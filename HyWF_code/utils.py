import numpy as np
import pickle
from pprint import pprint
import os
from constants import *
import glob
import re
import random
import time
import sys

num_Trees = 1000
n_jobs = 48


def save_and_print_results(results, args=None, name="", dont_print=False, only_save=False, dont_save_arg=False,
                           verbose_arg=INFO):

    if args is None or not args.dont_save:
        # write results in file
        sub_dir = "/".join(name.split("/")[:-1])
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        f_results = open(name + ".results", "wb")
        pickle.dump(results, f_results)
        f_results.close()

        if only_save:
            return

    avg_results = {k: np.mean(vals) for k, vals in results.items() if k != "f_names"}
    max_results = {k: max(vals) for k, vals in results.items() if k != "f_names"}

    if args is not None:
        verbose = args.verbose
        if dont_print:
            verbose = NONE
        dont_save = args.dont_save
    else:
        verbose = verbose_arg
        dont_save = dont_save_arg

    if "real_bytes_in" in results:
        if verbose <= INFO:
            print("Has %d file results" % len(results["real_bytes_in"]))
        avg_results["n_exp"] = len(results["real_bytes_in"])
        total_real = np.array(results["real_bytes_in"]) + np.array(results["real_bytes_out"])
        total_dummy = np.array(results["dummy_bytes_in"]) + np.array(results["dummy_bytes_out"])
        ind_max = np.argmax(np.divide(total_real + total_dummy, total_real))
        if verbose <= INFO:
            print("Max overhead is %f for %s" % (float(total_real[ind_max] + total_dummy[ind_max]) / total_real[ind_max],
                                               results["f_names"][ind_max]))
    elif "avg_interval" in results:
        if verbose <= INFO:
            print("Has %d file results" % len(results["avg_interval"]))
        avg_results["n_exp"] = len(results["avg_interval"])

    if verbose <= INFO:
        print("Average: ")
        pprint(avg_results)
        print("Max: ")
        pprint(max_results)

    f_stats = None
    if not dont_save:
        # write stats in file
        f_stats = open(name + ".stats", "w")
        f_stats.write("Average: " + repr(avg_results) + "\n")
        f_stats.write("Max: " + repr(max_results) + "\n")

    sum_real = avg_results["real_bytes_in"] + avg_results["real_bytes_out"]
    bytes_ov_str = \
        ("Bandwidth overhead ratio is in:%.2f" % ((avg_results["dummy_bytes_in"] + avg_results["real_bytes_in"])
                                                  / avg_results["real_bytes_in"])) + \
        ", " + ("out:%.2f" % ((avg_results["dummy_bytes_out"] + avg_results["real_bytes_out"])
                               / avg_results["real_bytes_out"])) + ", total " \
        + ("%.2f" % ((sum_real + avg_results["dummy_bytes_in"] + avg_results["dummy_bytes_out"]) / sum_real))
    if verbose <= INFO:
        print(bytes_ov_str)
    if f_stats:
        f_stats.write(bytes_ov_str + "\n")

    delay_ov_str = \
        "Latency overhead ratio is %.2f"  % (avg_results["max_time_after"]/ avg_results["max_time_before"])
    if verbose <= INFO:
        print(delay_ov_str)
    if f_stats:
        f_stats.write(delay_ov_str + "\n")

    if "-H" in name and "sec_channel_bytes_in" in avg_results:  # Hybrid
        sum_sec_channel = avg_results["sec_channel_bytes_in"]+avg_results["sec_channel_bytes_out"]
        channel_ov_str = \
            ("Sent in: %.2f" % (100 * avg_results["sec_channel_bytes_in"] / avg_results["real_bytes_in"])) + " %, " \
            + ("out: %.2f" % (100 * avg_results["sec_channel_bytes_out"] / avg_results["real_bytes_out"])) \
            + " % on secure channel, total " + ("%.2f" % (100*(sum_sec_channel/sum_real))) + " %"
        if verbose <= INFO:
            print(channel_ov_str)
        if f_stats:
            f_stats.write(channel_ov_str + "\n")

    if f_stats:
        f_stats.close()

    return avg_results


def find_delimiter(in_directory, with_indices=False):
    delimiter = ""
    for f in os.scandir(in_directory):
        f_name = in_directory + f.name
        if "-" in f_name or "_" in f_name:
            delimiter = "-" if "-" in f_name else "_"
            break

    if with_indices:
        list_files = [f_name.split("/")[-1] for f_name in glob.glob(in_directory + "*")]
        site_indices = list(set([int(f_name.split(delimiter)[0]) for f_name in list_files if delimiter in f_name]))
        return delimiter, site_indices

    return delimiter


def range_from_floats(str_array):
    if any([".." in x for x in str_array]):
        float_array = []
        for x in str_array:
            if ".." in x:
                toks = x.split("..")
                if len(toks) != 3:
                    print("If providing probas with range, must be start..step..end")
                    return []
                start = float(toks[0])
                end = float(toks[2])
                step = float(toks[1])
                tps = list(np.arange(start, end+step/2, step))
                float_array += tps
            else:
                float_array.append(float(x))
    else:
        float_array = list(map(float, str_array))

    float_array.sort()
    return float_array


def get_exp(avg_t, dist):
    if type(avg_t) is int and avg_t == 1:
        # no exponential if value is 1
        return 1
    # exponentional time
    if dist == "exp":
        return random.expovariate(float(1)/avg_t)
    else:  # fixed
        return avg_t


def check_data_format(train_directories, test_directory, light=False):
    print("Checking data format...")
    t_beg = time.time()
    # check formatting of directories
    train_directories = [d + ("/" if d[-1] != "/" else "") for d in train_directories]
    test_directory = test_directory + ("/" if test_directory[-1] != "/" else "")

    if light:
        return train_directories, test_directory, 0, 0, 0, 0, 0, 0, "", ""

    # determines first index (0 or 1) and last index

    suffix = ""
    delimiter = ""
    first_open_sitenum_per_dir = []
    first_closed_sitenum_per_dir = []
    first_closed_instnum_per_dir = []
    max_closed_sitenum_per_dir = []
    max_closed_instnum_per_dir = []
    max_open_sitenum_per_dir = []
    for d in list(set(train_directories+[test_directory])):
        print(d)
        # get pickle format if it exists
        if os.path.exists(d + "format.pickle"):
            format_res = pickle.load(open(d + "format.pickle", "rb"))
            first_open_sitenum_per_dir.append(format_res["first_open_sitenum"])
            first_closed_sitenum_per_dir.append(format_res["first_closed_sitenum"])
            first_closed_instnum_per_dir.append(format_res["first_closed_instnum"])
            max_closed_sitenum_per_dir.append(format_res["max_closed_sitenum"])
            max_closed_instnum_per_dir.append(format_res["max_closed_instnum"])
            max_open_sitenum_per_dir.append(format_res["max_open_sitenum"])
            delimiter = format_res["delimiter"]  # we assume that delimiter and suffix are same for all dirs
            suffix = format_res["suffix"]
            continue
        # otherwise compute it
        first_it = next(glob.iglob(d + "*,0*"), None)  # first check with index
        if first_it is None:
            it = glob.iglob(d + "*")
        else:
            it = glob.iglob(d + "*,0*")

        first_open_sitenum = 1000000
        first_closed_sitenum = 1000000
        first_closed_instnum = 1000000
        max_closed_sitenum = 0
        max_closed_instnum = 0
        max_open_sitenum = 0
        first_file = ""
        has_closed_world = False
        for f_name in it:
            if "pickle" in f_name:
                continue
            if first_file == "":
                first_file = f_name
            sname = f_name.split("/")[-1].split(".")[0]
            if "_" in sname or "-" in sname:
                delimiter = ("-" if "-" in sname else "_")
                has_closed_world = True
                toks = re.split("[_-]", sname)
                first_closed_sitenum = min(int(toks[0]), first_closed_sitenum)
                first_closed_instnum = min(int(toks[1].split(",")[0]), first_closed_instnum)
                max_closed_sitenum = max(int(toks[0]), max_closed_sitenum)
                max_closed_instnum = max(int(toks[1].split(",")[0]), max_closed_instnum)
            else:
                first_open_sitenum = min(int(sname.split(",")[0]),first_open_sitenum)
                max_open_sitenum = max(int(sname.split(",")[0]), max_open_sitenum)

        if first_open_sitenum == np.float("inf"):
            first_open_sitenum = 0

        max_closed_sitenum += (first_closed_sitenum == 0)
        max_closed_instnum += (first_closed_instnum == 0)
        if has_closed_world:
            max_open_sitenum += (first_open_sitenum == 0)
        else:
            max_open_sitenum = 0

        if "." in first_file.split("/")[-1]:
            suffix = "." + first_file.split(".")[-1]
        else:
            suffix = ""

        first_open_sitenum_per_dir.append(first_open_sitenum)
        first_closed_sitenum_per_dir.append(first_closed_sitenum)
        first_closed_instnum_per_dir.append(first_closed_instnum)
        max_closed_sitenum_per_dir.append(max_closed_sitenum)
        max_closed_instnum_per_dir.append(max_closed_instnum)
        max_open_sitenum_per_dir.append(max_open_sitenum)

        format_res = {"first_open_sitenum":first_open_sitenum, "first_closed_sitenum":first_closed_sitenum,
                      "first_closed_instnum":first_closed_instnum, "max_closed_sitenum":max_closed_sitenum,
                      "max_closed_instnum":max_closed_instnum, "max_open_sitenum":max_open_sitenum,
                      "suffix": suffix, "delimiter": delimiter}

        print("Saving format in pickle for dir %s" % d)
        pickle.dump(format_res, open(d + "format.pickle", "wb"))

    first_open_sitenum = min(first_open_sitenum_per_dir)
    first_closed_sitenum = min(first_closed_sitenum_per_dir)
    first_closed_instnum = min(first_closed_instnum_per_dir)
    max_closed_sitenum = max(max_closed_sitenum_per_dir)
    max_closed_instnum = max(max_closed_instnum_per_dir)
    max_open_sitenum = max(max_open_sitenum_per_dir)

    print("Done in %s seconds..." % repr(time.time() - t_beg))

    return train_directories, test_directory, max_closed_sitenum, max_closed_instnum, max_open_sitenum, \
           first_closed_sitenum, first_closed_instnum, first_open_sitenum, delimiter, suffix


def get_ts_size(x):
    x = x.replace("\n", "")
    x = re.split("[\t ]", x)
    if len(x) > 1:
        ts = float(x[0])
        size = int(float(x[1]))
    else:
        ts = -1
        size = int(float(x[0]))

    return ts, size