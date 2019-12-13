"""
The class Histogram provides an interface to generate and sample probability
distributions represented as histograms.
"""
import math
import operator
import random
from random import randint
from bisect import bisect_right
from scipy.stats import genpareto
import sys
from scipy.stats import norm
import numpy as np

from constants import *
import constants as ct

# shortcuts
from constants import INF

import logging


class Histogram:
    """Provides methods to generate and sample histograms of prob distributions."""

    logger = logging.getLogger('wtfpad')

    def __init__(self, hist, interpolate=True, remove_tokens=False, decay_by=0, name='', thr=2,
                 logger=logging.getLogger('wtfpad')):
        """Initialize an histogram.

        `hist` is a dictionary. The keys are labels of an histogram. They represent
        the value of the rightmost endpoint of the right-open interval of the bin. The
        dictionary value corresponding to that key is a count of tokens representing
        the frequency in the histogram.

        For example, the histogram:
         5-                                                     __
         4-                  __                                |  |
         3-        __       |  |                __             |  |
         2-       |  |      |  |               |  |            |  |
         1-       |  |      |  |               |  |            |  |
                [0, x_0) [x_0, x_1), ..., [x_(n-1), x_n), [x_n, infinite)

        Would be represented with the following dictionary:

            h = {'x_0': 3, 'x_1': 4, ..., 'x_n': 3, INF: 5}

        `interpolate` indicates whether the value is sampled uniformly
        from the interval defined by the bin (e.g., U([x_0, x_1)) or the value
        of the label is returned. In case of a discrete histogram we would have:

         5-                                                     __
         4-                  __                                |  |
         3-        __       |  |                __             |  |
         2-       |  |      |  |               |  |            |  |
         1-       |  |      |  |               |  |            |  |
                  x_0       x_1      ...       x_n           infinite

        `removeToks` indicates that every time a sample is drawn from the histrogram,
        we remove one token from the count (the values in the dictionary). We keep a
        copy of the initial value of the histogram to re assign it when the histogram
        runs out of tokens.

        We assume all values in the dictionary are positive integers and
        that there is at least a non-zero value. For efficiency, we assume the labels
        are floats that have been truncated up to some number of decimals. Normally, the
        labels will be seconds and since we want a precision of milliseconds, the float
        is truncated up to the 3rd decimal position with for example round(x_i, 3).
        """
        self.name = name
        self.hist = hist
        self.inf = False
        self.interpolate = interpolate
        self.remove_tokens = remove_tokens
        self.logger = logger
        self.thr = thr
        self.prob0 = 0

        if Histogram.is_empty(hist):
            self.interpolate = False
            self.remove_tokens = False

        # create template histogram
        self.template = dict(hist)
        self.avg_template = self.mean()
        self.logger.log(DEBUG, "Average value is %f", self.avg_template)

        # store labels in a list for fast search over keys
        self.labels = sorted(self.hist.keys())
        self.n = len(self.labels)

        # decay_by is the number of tokens we add to the infinity bin after
        # each successive padding packet is sent.
        self.decay_by = decay_by

        if INF in self.hist:
            self.prop_inf = float(self.hist[INF])/self.sum_noinf_toks()
        else:
            self.prop_inf = 0

        # dump initial histogram
        self.logger.log(VDEBUG, self.dump_histogram())

    def dump_histogram(self):
        """Print the values for the histogram."""
        s = "Dumping histogram: %s with %.3f INF, avg=%f\n" % (self.name, self.prop_inf, self.avg_template)
        if sum(self.hist.values()) > 3:
            s += "\tMean: %s\n" % self.mean()
            s += "\tVariance: %s\n" % self.variance()
        if self.interpolate:
            s += "\t[0, %s): %s; " % (self.labels[0], self.hist[self.labels[0]])
            for labeli, labeli1 in zip(self.labels[0:-1], self.labels[1:]):
                s += "[%s, %s): %s; " % (labeli, labeli1, self.hist[labeli1])
            s = s[:-1]  # remove last ;
            s += "\n"
        else:
            for label, count in self.hist.items():
                s += "(%s, %s); " % (label, count)
        return s

    def get_label_from_float(self, f):
        """Return the label for the interval to which `f` belongs."""
        if f in self.labels:
            return f
        return self.labels[bisect_right(self.labels, f)]

    def sum_noinf_toks(self):
        return sum([v for k, v in self.hist.items() if k != INF])

    def reset_infinite_bins(self):
        if self.prop_inf > 0:
            sum_noinf = self.sum_noinf_toks()
            self.hist[INF] = max(1, int(self.prop_inf*sum_noinf))

    def is_biased(self, f):
        if self.thr == 0:
            return False
        if f > 0.3*self.avg_template:
            # larger than average, is not biased (bias is only for small delays)
            self.logger.log(ALL, "Token %f is not biased (larger than average)" % f)
            return False
        ratios = {k: self.hist[k]/self.template[k] for k in self.template.keys() if self.template[k] != 0 and k != INF}
        label = self.get_label_from_float(f)
        if label not in ratios:
            # template[label] = 0
            #return True
            is_small = label < min(ratios.keys())
            if is_small:
                self.logger.log(ALL, "Token %f is biased (not in ratios, small)" % f)
            else:
                self.logger.log(ALL, "Token %f is not biased (not in ratios, large)" % f)
            return is_small  # return only if token is small
        ratios = list(ratios.values())
        avg_ratio = float(np.mean(ratios))
        std_ratio = float(np.std(ratios))
        if avg_ratio < 0.2:
            # too small to be meaningful
            self.logger.log(ALL, "Token %f is not biased (not enough values, avg=%f, std=%f)" % (f, avg_ratio, std_ratio))
            return False
        next_ratio = max(0,(self.hist[label] - 1)/self.template[label])
        if next_ratio < avg_ratio - self.thr*std_ratio:
            self.logger.log(ALL, "Token %f is biased (next_ratio=%f, avg=%f, std=%f, thr=%f)"
                            % (f, next_ratio, avg_ratio, std_ratio, self.thr))
            return True
        self.logger.log(ALL, "Token %f is not biased" % f)
        return False
        #return self.remove_token(f, only_check=True)

    def remove_token(self, f, padding=True, only_check=False):
        # TODO: move the if below to the calls to the function `remove_token`
        if self.remove_tokens:
            self.logger.log(VDEBUG, "[histo %s] Must remove sample %f", self.name, f)
            if padding:
                if INF in self.hist:
                    self.hist[INF] += self.decay_by
                #if INF in self.template:
                #    self.template[INF] += self.decay_by

            label = self.get_label_from_float(f)
            pos_counts = sorted([l for l in self.labels if self.hist[l] > 0])

            # else remove tokens from label or the next non-empty label on the left
            # if there is none, continue removing tokens on the right.
            if label not in pos_counts:
                #logger.debug("%s %s" % (pos_counts, self.hist))
                if label < pos_counts[0]:
                    label = pos_counts[0]
                    self.logger.log(VDEBUG, "[histo %s] Remove smallest token %s! Remaining tokens: %d/%d", self.name,
                                    label, self.hist[label]-1, sum(self.hist.values()))
                    too_small = True
                else:
                    test_label = label
                    #if max(pos_counts[:-1]) < label:
                    #    # token larger than all finite tokens, remove infinity
                    #    label = INF
                    #    #label = pos_counts[bisect_right(pos_counts, label) - 1]
                    #    too_small = False
                    #else:

                    # remove label to the left (largest smaller)
                    label = pos_counts[bisect_right(pos_counts, label) - 1]
                    if label > test_label:
                        # had to remove to the right (token too small)
                        too_small = True
                    else:
                        too_small = False
                    self.logger.log(VDEBUG, "[histo %s] Remove token %s instead of %s! Remaining tokens: %d/%d",
                                    self.name, label, test_label, self.hist[label]-1, sum(self.hist.values()))
                    # self.logger.log(ALL, "%s", self.dump_histogram())
            else:
                self.logger.log(VDEBUG, "[histo %s] Remove normal token %s! Remaining tokens: %d/%d", self.name,
                                label, self.hist[label]-1, sum(self.hist.values()))
                too_small = False

            if not only_check:
                self.hist[label] -= 1

            # if histogram is empty, refill the histogram
            if sum(self.hist.values()) == 0:
                self.refill_histogram()

            return too_small

    def mean(self):
        return sum([k * v for k, v in self.hist.items() if k != INF])/sum([v for k, v in self.hist.items() if k != INF])

    def variance(self):
        m = self.mean()
        n = sum([v for k, v in self.hist.items() if k != INF])
        if n < 2:
            raise ValueError("The sample is not big enough for an unbiased variance.")
        return sum([k * ((v - m) ** 2) for k, v in self.hist.items() if k != INF]) / (n - 1)

    def __str__(self):
        return self.dump_histogram()

    def refill_histogram(self):
        """Copy the template histo."""
        self.hist = dict(self.template)
        self.logger.log(DEBUG, "\n[histo] Refilled histogram: %s\n", self.name)

    def random_sample(self):
        """Draw and return a sample from the histogram."""
        self.logger.log(ALL, "[histo - %s] Draw random sample...", self.name)

        if self.prob0 > 0:
            # return 0 with some probability
            if random.random() < self.prob0:
                self.logger.log(ALL, "[histo - %s] Returns 0", self.name)
                return 0.0

        total_tokens = int(sum(self.hist.values()))
        prob = randint(1, total_tokens) if total_tokens > 0 else 0
        init_prob = prob
        for i, label_i in enumerate(self.labels):
            prob -= self.hist[label_i]
            if prob > 0:
                continue
            if not self.interpolate or i == self.n - 1:
                self.logger.log(ALL, "[histo - %s] Tokens = %s, prob = %s, p=%f", self.name,
                                sum(self.hist.values()), init_prob, label_i)
                return label_i
            label_i_1 = 0 if i == 0 else self.labels[i - 1]
            if label_i == INF:
                self.logger.log(ALL, "[histo - %s] Tokens = %s, prob = %s, labels=%f-%f, p=inf", self.name,
                                sum(self.hist.values()), init_prob, label_i_1, label_i)
                return INF
            p = label_i + (label_i_1 - label_i) * random.random()
            self.logger.log(ALL, "[histo - %s] Tokens = %s, prob = %s, labels=%f-%f, p=%f", self.name,
                            sum(self.hist.values()), init_prob, label_i_1, label_i, p)
            return p
        raise ValueError("In `histo.random_sample`: probability is larger than range of counts!")

    @classmethod
    def get_intervals_from_endpoints(self, ep_list):
        """Return list of intervals built from a list of endpoints."""
        return [[i, j] for i, j in zip(ep_list[:-1], ep_list[1:])]

    @classmethod
    def is_empty(self, d):
        return len(d.keys()) == 1 and INF in d.keys()

    @classmethod
    def divide_histogram(self, histogram, divide_by=None):
        if divide_by == None:
            return histogram, histogram
        if divide_by == 'mode':
            divide_by = max(histogram.    items(), key=operator.itemgetter(1))[0]
        high_bins = {k: v for k, v in histogram.items()  if k >= divide_by}
        low_bins = {k: v for k, v in histogram.items() if k <= divide_by}
        low_bins.update({INF: 0})
        high_bins.update({divide_by: 0})
        return low_bins, high_bins

    @classmethod
    def skew_histo_one_bin(self, d, side='left'):
        keys = sorted([k for k in d])
        if side == "left":
            pass
        elif side == "right":
            keys = keys[::-1]
        else:
            raise ValueError("No side %s." % side)
        assert(len(keys) > 2)
        for left_ep, left_ep_1 in zip(keys[:-1], keys[1:]):
            d[left_ep] = d[left_ep_1]
        d[keys[-1]] = 0
        return d

    @classmethod
    def skew_histo(self, d, nbins, side="left"):
        """Shift histo nbins to left/right."""
        if nbins == 0:
            return d
        for _ in range(nbins):
            d = Histogram.skew_histo_one_bin(d, side)
        return d

    @classmethod
    def get_dict_histo_from_list(self, l):
        import numpy as np
        counts, bins = np.histogram(l, bins=self.create_exponential_bins(a=0, b=10, n=20))
        d = dict(zip(list(bins) + [INF], [0] + list(counts) + [0]))
        d[0] = 0  # remove 0 iner-arrival times
        return d

    @classmethod
    def dict_from_list(self, l, num_samples=1000):
        import numpy as np
        counts, bins = np.histogram([math.ceil(v) for v in random.sample(l, num_samples)],
                                    bins=self.create_exponential_bins(a=0, b=10, n=20))
        d = dict(zip(list(bins) + [INF], [0] + list(counts) + [0]))
        d[0] = 0  # remove 0 iner-arrival times
        return d

    @classmethod
    def compute_new_norm(self, mu, sigma, percentile):
        mu_prime = norm.ppf(percentile, mu, sigma)
        if percentile == 0.5:
            sigma_prime = sigma
        else:
            pdf_mu_prime = norm.pdf(mu_prime, mu, sigma)
            sigma_prime = 1 / (math.sqrt(2 * math.pi) * pdf_mu_prime)
        return mu_prime, sigma_prime

    @classmethod
    def dict_from_distr(self, name, params, scale=1.0, num_samples=10000, bin_size=50, percentile=0.5, factor=1):

        mu, sigma = params
        if mu > 5:
            bins_hist = np.linspace(mu-2*sigma, mu+2*sigma, bin_size)
        else:
            bins_hist = self.create_exponential_bins(a=0, b=10, n=bin_size)

        if name == "histo":
            # just fit the values in a histogram
            values = params
            counts, bins = np.histogram(values, bins=bins_hist)
            # remove 0 values
            prob0 = float(counts[0])/sum(counts)
            self.prob0 = max(prob0,0.5)
            counts[0] = 0
            # set the number of samples to num_samples
            scale = float(sum(counts))/num_samples
            counts = (np.array(counts)/scale).astype(int)
        elif name == "weibull":
            shape = params
            counts, bins = np.histogram(np.random.weibull(shape, num_samples) * scale,
                                        bins=bins_hist)

        elif name == "beta":
            a, b = params
            counts, bins = np.histogram(np.random.beta(a, b, num_samples) * scale,
                                        bins=bins_hist)

        elif name == "logis":
            location, scale = params
            counts, bins = np.histogram(np.random.logistic(location, scale, num_samples),
                                        bins=bins_hist)

        elif name == "lnorm":
            mu, sigma = params
            counts, bins = np.histogram(np.random.lognormal(mu, sigma, num_samples),
                                        bins=bins_hist)

        elif name == "norm":
            mu, sigma = params
            mu1, sigma1 = self.compute_new_norm(mu, sigma, percentile)
            self.logger.log(DEBUG, "New norm has mean=%f, std=%f (was mean=%f, std=%f)", mu1, sigma1, mu, sigma)
            counts, bins = np.histogram([s for s in np.random.normal(mu1, sigma1, num_samples) if s > 0],
                                        bins=bins_hist)

        elif name == "l10norm":
            # log10 is normal distribution with mean mu and std dev sigma
            mu2 = None
            prop_zero = None
            params = tuple(params)
            if len(params) == 2:
                # only mean and std dev
                mu, sigma = params
            elif len(params) == 3:
                # mean, std dev and proportion of zero
                mu, sigma, prop_zero = params
            elif len(params) == 4:
                # two distributions
                mu, sigma, mu2, sigma2 = params
            elif len(params) == 5:
                # two distributions and proportion of zero
                mu, sigma, mu2, sigma2, prop_zero = params

            mu_p, sigma_p = self.compute_new_norm(mu, sigma, percentile)
            self.logger.log(DEBUG, "New l10norm has mean=%f, std=%f (was mean=%f, std=%f)", mu_p, sigma_p, mu, sigma)

            counts, bins = np.histogram([s for s in np.power(10, np.random.normal(mu_p, sigma_p, num_samples)) if s > 0],
                                        bins=bins_hist)

            if mu2 is not None:
                mu2_p, sigma2_p = self.compute_new_norm(mu2, sigma2, percentile)
                self.logger.log(DEBUG, "New l10norm-2 has mean=%f, std=%f (was mean=%f, std=%f)",
                                mu2_p, sigma2_p, mu2, sigma2)
                counts2, bins2 = np.histogram(
                    [s for s in np.power(10, np.random.normal(mu2_p, sigma2_p, num_samples)) if s > 0],
                    bins=bins_hist)
                assert np.equal(bins, bins2).all()
                counts = (np.sum([counts, counts2], axis=0)/2).astype(int)

            if prop_zero is not None:
                add_zero = int(prop_zero*num_samples/(1-prop_zero))
                counts[0] += add_zero
                counts = (num_samples*counts/sum(counts)).astype(int)

        elif name == "gamma":
            shape, scale = params
            counts, bins = np.histogram(np.random.gamma(shape, scale, num_samples),
                                        bins=bins_hist)

        elif name == "pareto":
            shape, scale = params
            counts, bins = np.histogram(genpareto.rvs(shape, scale=scale, size=num_samples),
                                        bins=bins_hist)

        elif name == "empty":
            return NO_SEND_HISTO

        else:
            raise ValueError("Unknown probability distribution.")

        d = dict(zip(list(bins) + [INF], [0] + list(counts) + [0]))
        d[0] = 0  # remove 0 iner-arrival times

        if factor != 1:
            # multiply each inter-arrival time by factor
            d = {factor*k: v for k,v in d.items()}

        return d

    @classmethod
    def create_exponential_bins(self, sample=None, min_bin=None,
                                a=None, b=None, n=None):
        """Return a partition of the interval [a, b] with n number of bins.

        Alternatively, it can take a sample of the data and extract the interval
        endpoints by taking the minimum and the maximum.
        """
        if sample:
            a = min(sample)
            b = max(sample)
            if not min_bin:
                n = 20  # TODO: what is the best number of bins?
            n = int(b - a / min_bin)
        return ([b] + [(b - a) / 2.0 ** k for k in range(1, n)] + [a])[::-1]

    @classmethod
    def drop_first_n_bins(self, h, n):
        for k in sorted(h.keys())[:n]:
            del h[k]
        return h


def uniform(x):
    return new({x: 1}, interpolate=False, remove_tokens=False)


# Alias class name in order to provide a more intuitive API.
new = Histogram

