import numpy as np
from scipy import stats
import argparse
import glob
import re
import sys
from pprint import pprint
from utils import *


def compute_accuracy_args(args):

    args.train_probas = range_from_floats(args.train_probas)

    train_probas = list(map(lambda x: int(round(100*x)), args.train_probas))

    test_str = str(round(int(100 * args.test_proba))).zfill(2)
    all_files_test = sorted(glob.glob(args.directory+"/flearner*-"+test_str+"-*"))

    filenames = []
    for f in all_files_test:
        train_ind = int(f.split("/")[-1].split(".")[0].split("-")[-1])

        if train_ind in train_probas:
            filenames.append(f)

    train_str = " ".join(map(str,args.train_probas))
    if args.verbose:
        print("Compute accuracy for p_test=" + str(args.test_proba)+", p_train="+train_str)
        print("Files are %s" % " ".join(filenames))

    return compute_accuracy(filenames, args.open_world, verbose=args.verbose, sliding_window=args.sliding_window)


def compute_accuracy(filenames, open_world, twoclass=True, verbose=False, sliding_window=None):
    # twoclass valid only for open_world

    done_line = ""
    if sliding_window is None or sliding_window == 0:
        results = {}
        for filename in filenames:
            f = open(filename, "r")
            lines = f.readlines()
            f.close()

            # compute accuracy
            for line in lines:
                if "p_train" in line or "feature" in line:
                    continue
                if "done" in line:
                    done_line = line
                    continue
                tokens = line.strip().split()
                exp = tokens[0]
                if exp not in results:
                    results[exp] = []
                values = list(map(float, tokens[1:]))
                guessed_site = np.argmax(values)
                if open_world:
                    if guessed_site == len(values) - 1:
                        # guessed white list
                        guessed_site = -1
                    elif twoclass:
                        # guessed black list
                        guessed_site = 0
                results[exp].append(guessed_site)
    else:
        print("Not supported")
        return [], [], []
    #     # this is a quick and ugly way to do TODO: do better
    #     results = {}
    #     counts = {}
    #     for p in map(lambda x:round(x,1), np.arange(0.1, 1.01, 0.1)):
    #         min_p = max(0, p-sliding_window)
    #         max_p = min(1, p+sliding_window)
    #         results_loc = {}
    #         #print("Learning for p="+str(p))
    #         for filename in filenames:
    #             f = open(filename, "r")
    #             lines = f.readlines()
    #             f.close()
    #
    #             # compute accuracy
    #             for line in lines:
    #                 tokens = line.strip().split()
    #                 if "p_train" in line or "feature" in line or "done" in line:
    #                     if tokens[0] == "p_train":
    #                         p_train = float(tokens[2].replace(",",""))
    #                         if p_train < min_p or p_train > max_p:
    #                             break
    #                         #print("Look at file " + filename)
    #                     continue
    #                 exp = tokens[0]
    #                 if exp not in results_loc:
    #                     results_loc[exp] = []
    #                 values = list(map(float, tokens[1:]))
    #                 guessed_site = np.argmax(values)
    #                 if open_world:
    #                     if guessed_site == len(values) - 1:
    #                         # guessed white list
    #                         guessed_site = -1
    #                     elif twoclass:
    #                         # guessed black list
    #                         guessed_site = 0
    #                 results_loc[exp].append(guessed_site)
    #
    #         for exp in results_loc:
    #             if len(results_loc[exp]) == 0:
    #                 continue
    #             #print("Results for %s are %s" % (exp, repr(results_loc[exp])))
    #             if hasattr(stats.mode(results_loc), "mode"):
    #                 # python 3
    #                 guessed_site = stats.mode(results_loc[exp]).mode[0]  # most recurring value (mode)
    #             else:
    #                 # python 2
    #                 guessed_site = stats.mode(results_loc[exp])[0][0]
    #
    #             counts_loc = len([x for x in results_loc[exp] if x == guessed_site])
    #             #print("Guessed %d with count %d" % (guessed_site, counts_loc))
    #             if exp not in counts or counts[exp] < counts_loc:
    #                 counts[exp] = counts_loc
    #                 results[exp] = [guessed_site]
    #                 #print("New guessed site for %s is %d with count %d" % (exp, guessed_site, counts_loc))

    #pprint(results)

    tp = 0
    fp = 0
    monitored = 0
    unmonitored = 0
    print("Has %d results" % len(results))
    for exp in results:
        toks = re.split("_", exp)
        if len(toks) > 0 and int(toks[0]) >= 0:  # foreground page
            if open_world and twoclass:
                true_site = 0
            else:
                true_site = int(toks[0])
        else: # background page
            true_site = -1
            if not open_world:
                if verbose:
                    print("WARNING: arg is not open_world whereas exp is")
                return -1,-1,-1
        if hasattr(stats.mode(results[exp]),"mode"):
            # python 3
            guessed_site = stats.mode(results[exp]).mode[0]  # most recurring value (mode)
        else:
            # python 2
            guessed_site = stats.mode(results[exp])[0][0]
        counts_loc = len([x for x in results[exp] if x == guessed_site])
        #print("New guessed site for %s is %d with count %d" % (exp, guessed_site, counts_loc))
        if open_world:
            if true_site >= 0:
                monitored += 1
                if true_site == guessed_site:
                    tp += 1
            else:
                unmonitored += 1
                if true_site != guessed_site:
                    # guessed black list whereas white list
                    fp += 1
        else:
            monitored += 1
            if guessed_site == true_site:
                tp += 1

    if open_world and unmonitored == 0:
        print("No unmonitored whereas open-world")
        return -1, -1, -1

    if monitored > 0:
        tpr = float(tp) / float(monitored)

        if verbose:
            print("tp:" + str(tp) + (", fp:" + str(fp) if open_world else "")+
                  ", monitored:" + str(monitored)+(", unmonitored:"+str(unmonitored) if open_world else ""))
            print("TPR is " + ("%.2f" % (100*tpr))+" %")
        if open_world:
            fpr = float(fp) / float(unmonitored)
            if verbose:
                print("FPR is " + ("%.2f" % (100*fpr))+" %")
            # bayesian detection rate (see Hayes et al.)
            #bdr = tpr * monitored / (tpr * monitored + fpr * unmonitored)
            #print("BDR is " + ("%.2f" % (100*bdr))+" %")
        else:
            fpr = 0
    else:
        tpr = -1
        fpr = -1

    if done_line != "":
        if verbose:
            print(done_line)
        tokens = done_line.strip().split(" ")
        duration = float(tokens[tokens.index("seconds")-1])
    else:
        duration = -1

    return tpr, fpr, duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory", help="Directory with results")
    parser.add_argument("-te", "--test_proba", type=float, help="Test probabilities")
    parser.add_argument("-tr", "--train_probas", nargs="+", help="Train probabilities to test")
    parser.add_argument("-o", "--open_world", action="store_true", help="If open_world (blacklist/whitelist)")
    parser.add_argument("-v", "--verbose", action="store_true", help="If verbose")

    args = parser.parse_args()

    compute_accuracy_args(args)