import numpy as np
from math import sqrt, pi, ceil
from scipy.stats import norm
from bisect import insort_left
import random
import sys
import histograms as histo
from parser import Flow, Packet, Trace, Machine
import os
# shortcuts
from constants import *
import os,inspect
import logging
import matplotlib.pyplot as plt
import scipy.stats
import pickle

# shortcut
ht = histo.Histogram


class AdaptiveSimulator(object):
    """Simulates adaptive padding's original design on real web data."""

    def __init__(self, config):
        # parse arguments
        self.interpolate = bool(config.get('interpolate', True))
        self.remove_tokens = bool(config.get('remove_tokens', True))
        self.stop_on_real = config.stop_on_real
        self.percentile = config.percentile
        self.percentile_burst = config.percentile_burst
        self.logger = config.logger
        ht.logger = config.logger
        self.packet_sent = 0
        self.hybrid = config.hybrid
        self.thr = config.thr
        #self.thr_burst = config.thr_burst
        #self.prob_burst = config.prob_burst
        self.max_consecutive = INF
        self.folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.plotted = 0
        self.website = config.website
        self.stop_on_last = False
        self.flows = {IN: Flow(IN), OUT: Flow(OUT)}
        self.min_bursts = {}

        self.factor_interarrivals = {IN: 1, OUT: 1}

        # set seeds
        random.seed(config.seed)
        np.random.seed(config.seed)

        # TODO: different number of state machines per direction?
        self.num_state_machines = 0  # defined at configuration

        # the distribution of packet lengths is fixed in Tor
        self.length_distrib = histo.uniform(config.size_packets)
        self.size_packets = config.size_packets

        # has_rcv: whereas send packet upon reception (depends on whether a rcv distribution exists)
        self.has_rcv = False

        self.all_hists = []
        # initialize dictionary of distributions
        if "cluster_path" in config:
            self.hist = self.initialize_distributions_clusters(config["cluster_path"])
        else:
            distributions = {k: v for k, v in config.items() if 'dist' in k}
            self.hist = self.initialize_distributions(distributions)

    def refill_all_hists(self):
        for hist in self.all_hists:
            hist.refill_histogram()

    def reset_all_infinity_bins(self):
        for hist in self.all_hists:
            hist.reset_infinite_bins()

    def reinitialize(self, config):
        self.__init__(config)

    def handle_packet(self, packet, i, flows, send_to_secure, next_rand_out, trace, count=0):

        if not flows[IN].done or not flows[OUT].done:
            if "rand_out_dist" in self.hist and (next_rand_out == -1 or packet.timestamp == next_rand_out):
                next_rand_out = self.hist["rand_out_dist"].random_sample()
                self.hist["rand_out_dist"].remove_token(next_rand_out)
                dummy = self.generate_dummy(packet, flows[OUT], next_rand_out, rand_out=True)
                insort_left(trace, dummy)
                next_rand_out += packet.timestamp
                self.logger.log(VDEBUG, "Send a dummy random outgoing packet %s", dummy.formatted())

        # flow in the direction of the packet and the opposite
        flow = flows[packet.direction]
        oppflow = flows[-packet.direction]  # opposite direction

        # initialize both flows
        if i == 0:
            if self.num_state_machines == 2:
                flow.state = BURST
                oppflow.state = BURST
            else:
                flow.state = GAP
                oppflow.state = GAP

        if not flow.initialized and not packet.dummy:
            self.logger.log(DEBUG, "Initialize flow %s", flow.name)
            flow.initialized = True
            flow.done = False

        self.logger.log(VDEBUG, "\nGot packet #%d %s for flow %s, state %s (%d)", self.packet_sent,
                        packet.formatted(), flow.name, DICT_STATES[flow.state], flow.state)

        old_state = flow.state
        # update state
        if i > 0:
            self.update_state(packet, flow)

        if not packet.sec_channel_pre:
            # run adaptive padding in the flow direction, check whether to send packet to secure channel
            if self.add_padding(i, trace, flow, 'snd') and count < self.max_consecutive:
                # remove packet
                ind = self.get_next_packet(trace, i, flow.direction)
                self.logger.log(VDEBUG, "Send packet %s to secure channel (next is %s)", trace[ind].formatted(),
                                trace[i + 1].formatted() if len(trace) > i + 1 else "None")
                assert (not trace[ind].dummy)
                assert (not trace[ind].sec_channel)
                assert (not trace[ind].sec_channel_pre)
                send_to_secure.append(trace[ind])
                # packets are not removed here to participate in opposite padding
                trace[ind].sec_channel = True
                #popped = trace.pop(ind)

                # restore state (as if nothing had happened)
                flow.state = old_state

                return self.handle_packet(packet, i, flows, send_to_secure, next_rand_out, trace, count+1)
        else:
            assert (not packet.dummy)
            send_to_secure.append(packet)
            self.logger.log(VDEBUG, "Packet %s is sent on secure channel", packet.formatted())

        if not packet.dummy and not packet.sec_channel and not packet.sec_channel_pre and flow.state == BURST:
            flow.current_burst += 1
            self.logger.log(ALL, "Increment counter of flow %s for packet %s, now %d", flow.name,
                            packet.formatted(), flow.current_burst)

        # run adaptive padding in the opposite direction,
        # as if the packet was received at the other side
        if self.has_rcv and (not flows[IN].done or not flows[OUT].done):
            if flow.timeout != INF:
                self.add_padding(i, trace, oppflow, 'rcv')

    def simulate(self, trace, prob_changes={}):
        """Adaptive padding simulation of a trace."""

        # initialize flows
        self.flows = {IN: Flow(IN), OUT: Flow(OUT)}
        for d in self.min_bursts:
            self.flows[d].min_burst = self.min_bursts[d]

        next_rand_out = -1

        num_packets = trace.total_size()

        send_to_secure = Trace()

        for i, packet in enumerate(trace):
            if packet.direction in prob_changes and len(prob_changes[packet.direction]) > 0 and\
                    packet == prob_changes[packet.direction][0][0]:
                # change factor for interarrivals
                self.factor_interarrivals[packet.direction] = 1 / prob_changes[packet.direction][0][1]
            self.packet_sent += 1
            packet.post_id = self.packet_sent

            self.handle_packet(packet, i, self.flows, send_to_secure, next_rand_out, trace)

            # pad packet length
            #packet.length = int(self.length_distrib.random_sample())

        trace = Trace([p for p in trace if not p.sec_channel and not p.sec_channel_pre])  # extract packets sent on secure channel

        # sort race by timestamp
        trace.sort(key=lambda x: x.timestamp)

        if num_packets > 0:
            self.logger.log(DEBUG, "Trace had %d packets, sent %d to secure, added %d (overheads sec:%f, dummy:%f)",
                            num_packets, len(send_to_secure), len(trace) + len(send_to_secure) - num_packets,
                            float(len(send_to_secure))/num_packets, float(len(trace) + len(send_to_secure))/num_packets)
        else:
            self.logger.log(DEBUG, "Trace had no packet")

        return trace, send_to_secure

    def add_padding(self, i, trace, flow, on):
        """Generate a dummy packet."""

        packet = trace[i]

        if flow.state == WAIT:
            return False

        timeout = INF
        histogram = self.hist[flow.state][flow.direction][on]
        if histogram is not None:
            timeout = histogram.random_sample()*self.factor_interarrivals[flow.direction]

        self.logger.log(ALL, "Flow %s: New timeout %f, state %s, %s, %s",
                        flow.name, timeout, DICT_STATES[flow.state], on, DIRS2EP[flow.direction])

        try:
            iat, next_is_dummy, next_is_sec = self.get_iat(i, trace, flow)
        except IndexError:
            self.logger.log(ALL, "Timestamp difference is INF")
            iat = INF
            if not flow.done:
                self.logger.log(DEBUG, "Flow %s is done", flow.name)
                flow.done = True
            next_is_dummy = True
            next_is_sec = False
            if flow.initialized:
                self.pad_end_flow(flow, trace)
            if self.stop_on_last:
                return False
            #histogram.reset_infinite_bins()
            #return False

        if iat == INF and timeout == INF:
            return False

        send_dummy = False
        # if iat <= 0 we do not have space for a dummy
        flow.timeout = timeout
        flow.expired = False
        old_iat = iat
        if iat > 0:
            if timeout < iat:

                send_dummy = True
                # timeout has expired
                flow.expired = True
                flow.timeout = timeout

                # the timeout has expired, we send a dummy packet
                dummy = self.generate_dummy(packet, flow, timeout)

                self.logger.log(VDEBUG, "Flow %s: Send a dummy packet %s on direction %d, state %s",
                                flow.name, dummy.formatted(), flow.direction,
                                DICT_STATES[flow.state])

                # correct the timeout
                iat = timeout

                # add dummy to trace
                insort_left(trace, dummy)

        if timeout >= old_iat and self.hybrid and not packet.dummy and not next_is_dummy and not next_is_sec:
            # next one is a real packet, check whether we send it on the secure channel
            if self.percentile_burst > 0 and flow.current_burst > flow.token_burst:
                # burst is too long, send on secure channel to break it
                self.logger.log(VDEBUG, "Burst too long (%d > %d) on flow %s, send next packet to secure channel",
                                flow.current_burst, flow.token_burst, flow.name)
                flow.current_sent_sec_burst += 1
                return True
            histogram_gap = self.hist[GAP][flow.direction][on]
            if histogram_gap is not None and old_iat > 0:
                is_biased = histogram_gap.is_biased(old_iat)
                if is_biased:
                    # means that we cannot remove this token from histogram (too small)
                    flow.current_sent_sec_bias += 1
                    return True

        if iat > 0:
            # remove the token from histogram gap (intra-burst arrival times)
            if histogram == self.hist[GAP][flow.direction][on] or not send_dummy:
                # this is an intra-burst inter-arrival time, remove token
                histogram = self.hist[GAP][flow.direction][on]
                if histogram is not None:
                    histogram.remove_token(iat/self.factor_interarrivals[flow.direction])
            #if histogram is not None:
            #    histogram.remove_token(iat)

        return False

    def init_burst(self, flow):
        if flow.current_burst > flow.token_burst:
            # burst is too long, send on secure channel to break it
            self.logger.log(VDEBUG, "Burst too long (%d > %d) on flow %s",
                            flow.current_burst, flow.token_burst, flow.name)
        burst_len = int(flow.current_burst)
        flow.burst_lengths.append(burst_len)
        flow.current_burst = 0
        old_token_burst = flow.token_burst
        if "burst_length_dist" in self.hist and flow.min_burst != INF:
            flow.token_burst = 0
            while flow.token_burst <= flow.min_burst:
                # do not draw smaller than min_burst
                flow.token_burst = self.hist["burst_length_dist"][flow.direction].random_sample()
        self.logger.log(DEBUG, "Burst length was %d (%f) on flow %s, has sent %d+%d on secure channel. "
                               "Draw new sample %f", burst_len, old_token_burst, flow.name, flow.current_sent_sec_burst,
                        flow.current_sent_sec_bias, flow.token_burst)
        flow.current_sent_sec_burst = 0
        flow.current_sent_sec_bias = 0

    def update_state(self, packet, flow):
        """Switch state accordingly to AP machine state."""

        old_state = flow.state

        if flow.state == WAIT and not packet.dummy:
            self.logger.log(ALL, "Real packet: burst")
            flow.state = BURST

        elif flow.state == BURST and flow.expired:
            self.logger.log(ALL, "Flow %s expired: gap", flow.name)
            flow.state = GAP

        elif flow.state == BURST and flow.timeout == INF:
            self.logger.log(ALL, "Timeout inf: wait")
            flow.state = WAIT

        elif flow.state == GAP and flow.timeout == INF:
            self.logger.log(ALL, "Timeout inf: burst")
            flow.state = BURST

        elif flow.state == GAP and not packet.dummy:
            if self.stop_on_real:
                self.logger.log(ALL, "Real packet (stop on real): wait")
                flow.state = WAIT
            else:
                self.logger.log(ALL, "Real packet: burst")
                flow.state = BURST

        if self.num_state_machines == 1:
            # with one state machine, burst is gap
            if flow.state == BURST:
                flow.state = GAP

        if old_state != flow.state:
            if old_state == BURST:
                self.init_burst(flow)
            self.logger.log(ALL, "Machine = %s, change state from %s to %s",
                            flow.name, DICT_STATES[old_state], DICT_STATES[flow.state])

    def get_iat(self, i, trace, flow):
        """Find previous and following packets to substract their timestamps."""
        self.logger.log(ALL, "Get timestamp difference on flow %s", flow.name)
        packet_0 = trace[i]
        packet_1 = trace[self.get_next_packet(trace, i, flow.direction)]
        diff_ts = packet_1.timestamp - packet_0.timestamp
        self.logger.log(ALL, "Timestamp diff %s, %s is %f",
                        packet_0.formatted(), packet_1.formatted(), diff_ts)
        return diff_ts, packet_1.dummy, packet_1.sec_channel_pre

    def get_next_packet(self, trace, i, direction):
        """Get the packet following the packet in position i with the same
        direction.
        """
        return trace.get_next_by_direction(i, direction)

    def pad_end_flow(self, flow, trace):
        # AP is supposed to run continuously. So, it cannot be fairly evaluated
        # with other classifiers f we implement this funciton.
        # TODO
        self.logger.log(ALL, "Pad end of flow %s", flow.name)
        self.init_burst(flow)

    def generate_dummy(self, packet, flow, timeout, rand_out=False):
        """Set properties for dummy packet."""
        ts = packet.timestamp + timeout
        #size_packet = int(self.length_distrib.random_sample())
        size_packet = self.size_packets
        return Packet(ts, flow.direction, size_packet, dummy=True, rand_out=rand_out)

    def sum_noinf_toks(self, h):
        return sum([v for k, v in h.items() if k != INF])

    ############  INITIALIZATION METHODS #################

    def load_histo(self, histo_fpath):
        with open(histo_fpath) as fi:
            tss = list(map(float, fi.readlines()))
        d = ht.dict_from_distr("histo", tss, bin_size=30)
        if self.plotted == -1:
            print(float(len([t for t in tss if t==0.0]))/len(tss))
            print(d)
            plt.hist(tss, bins=[0]+np.logspace(-8,2))
            plt.xscale("log")
            plt.show()
            sys.exit(0)
        self.plotted += 1
        return d

    def load_and_fit(self, histo_fpath, percentile=0.5, fit_distr='l10norm'):
        with open(histo_fpath) as fi:
            tss = list(map(float, fi.readlines()))
        log_tss = [np.log10(ts) for ts in tss if ts > 0]
        mu = np.mean(log_tss)
        sigma = np.std(log_tss)
        mu_prime = norm.ppf(percentile, mu, sigma)
        if percentile == 0.5:
            sigma_prime = sigma
        elif percentile < 0.5:
            pdf_mu_prime = norm.pdf(mu_prime, mu, sigma)
            sigma_prime = 1 / (sqrt(2 * pi) * pdf_mu_prime)
        else:
            raise ValueError("Skewing distrib toward longer inter-arrival times makes fake bursts distinguishable from real.")
        if self.plotted == -1:
            x = np.linspace(-8,2, 100)
            print(x)
            hist, edges = np.histogram(log_tss, bins=x)
            plt.plot(x[:-1], hist)
            print(log_tss[0:50])
            print(hist, edges)
            print("Mean=%f, std=%f, # samples %d" % (mu_prime, sigma_prime, len(log_tss)))
            hist2, edges2 = np.histogram([s for s in np.random.normal(mu_prime, sigma_prime, len(log_tss))], bins=x)
            plt.plot(edges2[:-1], hist2)
            #plt.xscale("log")
            plt.show()
            sys.exit(0)

        return ht.dict_from_distr(fit_distr, (mu_prime, sigma_prime), bin_size=30)

    def init_distrib(self, name, config_dist, drop=0, skew=0, percentile=None):
        # parse distributions parameters
        self.logger.log(VDEBUG, "Configuration of distribution \'%s\': %s (drop=%d, skew=%d)"
                        % (name, config_dist, drop, skew))
        dist, params = config_dist.split(',', 1)

        if percentile is None:
            percentile = self.percentile

        if dist == 'histo':
            param_split = params.split(',')
            if len(param_split) == 1:
                inf_config = -1
                histo_fpath = param_split[0]
            else:  # = 2
                inf_config = param_split[0]
                histo_fpath = param_split[1]
            histo_fpath = self.folder_path+"/"+histo_fpath.strip()
            if inf_config != -1:
                inf_config = float(inf_config.strip())
            self.logger.log(DEBUG, "Loading and fitting histogram from: %s" % histo_fpath)
            d = self.load_and_fit(histo_fpath, percentile=percentile)
            #d = self.load_histo(histo_fpath)
            if sum(d.values()) == 0:
                print("Histogram %s has no value" % histo_fpath)
                sys.exit(1)
            d = self.set_infinity_bin(d, name, inf_config)

        else:
            if len(params.split(',')) == 2:
                dist_params = params
                inf_config = -1
            else:
                inf_config, dist_params = params.split(',', 1)
                inf_config = float(inf_config.strip())
            dist_params = list(map(float, [x.strip() for x in dist_params.split(',')]))
            d = ht.dict_from_distr(name=dist, params=dist_params, bin_size=30, percentile=percentile)
            d = self.set_infinity_bin(d, name, inf_config)

        # drop first `drop` bins
        if drop > 0:
            d = ht.drop_first_n_bins(d, drop)

        # skew histograms
        if skew > 0:
            d = ht.skew_histo(d, skew)

        return d

    def initialize_distributions_clusters(self, cluster_path):
        cluster_path = self.folder_path+"/"+cluster_path.strip()
        cluster_pickle = pickle.load(open(cluster_path, "rb"))
        if self.website in cluster_pickle["point2clusters"]:
            c = cluster_pickle["point2clusters"][self.website]
        else:  # for open world websites, always use cluster 0
            c = 0
        self.logger.log(DEBUG, "Initialize distributions for website %s, member of cluster %d", self.website, c)
        distributions = {dist: ("norm,%f,%f" % (vals[0], vals[1])) if "burst_length_dist" in dist
                else ("l10norm,%d,%f,%f" % (vals[0], vals[1], vals[2])) for dist, vals in cluster_pickle[c].items()}

        return self.initialize_distributions(distributions)

    def initialize_distributions(self, distributions):
        hist = {}
        for k, v in distributions.items():
            if k == "rand_out_dist":
                self.logger.log(DEBUG, "\nCreate distribution for %s ", k)
                hist[k] = histo.new(self.init_distrib(k, v), self.interpolate, self.remove_tokens,
                                    name=k, logger=self.logger, thr=self.thr)
                self.all_hists.append(hist[k])
            elif "_burst_length_dist" in k:
                if self.percentile_burst > 0:
                    self.logger.log(DEBUG, "\nCreate distribution for %s ", k)
                    if "burst_length_dist" not in hist:
                        hist["burst_length_dist"] = {}
                    endpoint = k.replace("_burst_length_dist", "")
                    d = EP2DIRS[endpoint]
                    #self.min_bursts[d] = float(v.split(",")[1])  # min burst size to send on secure channel is the mean of distribution
                    hist["burst_length_dist"][d] = histo.new(self.init_distrib(k, v, percentile=self.percentile_burst),
                                                             self.interpolate, self.remove_tokens, name=k,
                                                             logger=self.logger, thr=self.thr)
                    self.all_hists.append(hist["burst_length_dist"][d])
                    self.min_bursts[d] = hist["burst_length_dist"][d].mean()
            else:
                toks = k.split('_')
                if len(toks) == 4:
                    # distribution with two state machines
                    endpoint, on, mode, _ = toks
                    if self.num_state_machines == 0:
                        self.num_state_machines = 2
                elif len(toks) == 3:
                    # distribution with one state machine
                    endpoint, on, _ = toks
                    mode = "gap"
                    if self.num_state_machines == 0:
                        self.num_state_machines = 1
                s = MODE2STATE[mode]
                if s not in hist:
                    hist[s] = {}
                d = EP2DIRS[endpoint]
                if d not in hist[s]:
                    hist[s][d] = {}
                if on == "rcv":
                    self.has_rcv = True
                self.logger.log(DEBUG, "\nCreate distribution for %s (%d), %s (%d), %s with %d sm",
                                mode, s, endpoint, d, on, self.num_state_machines)
                hist[s][d][on] = histo.new(self.init_distrib(k, v), self.interpolate, self.remove_tokens,
                                           name=k, logger=self.logger, thr=self.thr)
                self.all_hists.append(hist[s][d][on])

        return hist

    def set_infinity_bin(self, distrib, name, inf_config):
        '''Setting the histograms' infinity bins.'''
        if inf_config < 0:
            # do not set infinity bin, stop on last packet
            self.stop_on_last = True
            return distrib

        assert len(distrib.keys()) > 1
        # GAPS
        # we want the expectation of the geometric distribution of consecutive
        # samples from histogram to be the average number of packets in a burst.

        # Therefore, the probability of falling into the inf bin should be:
        # p = 1/N, (note the variance is going to be high)
        # where N is the length of the burst in packets.

        # Then, the tokens in infinity value should be:
        #  p = #tokens in inf bin / #total tokens <=>
        #  #tokens in inf bin = #tokens in other bins / (N - 1)

        if 'gap' in name:
            burst_length = int(inf_config)
            other_toks = self.sum_noinf_toks(distrib)
            distrib[INF] = ceil(other_toks / (burst_length - 1))

        # BURSTS
        # IN (server)
        # 5% of the time we sample from inf bin
        # (95% of burst will be followed by a fake burst)
        #
        # OUT (client)
        # 10% of the time we sample from inf bin
        # (90% of burst will be followed by a fake burst)
        # less padding in the direction from client to server because there is
        # also less volume.
        else:
            prob_burst = inf_config
            other_toks = self.sum_noinf_toks(distrib)
            distrib[INF] = ceil(other_toks / prob_burst)

        return distrib
