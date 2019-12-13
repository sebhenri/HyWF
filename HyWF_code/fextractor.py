import types
import math
from utils import *
from multiprocessing import Pool
from code_kfp import extract_Hayes
from code_cumul import extract_Panchenko
from code_knn import extract_Wang

# This file extracts the features for the different attacks (parallel version)
# TODO: one file for parallel and single thread


def write_features(directory, fname, p_in, p_out, ffiles_loc, type, args, fout, is_train, original_probas):

    if "." in fname:
        ffile_name = fout + "f"
    else:
        ffile_name = fout + ".f"

    if not os.path.exists(directory + fname):
        #print("\nWARNING: file " + fname + " does not exist")
        return [], [], 0, 0

    f = open(directory + fname, "r")

    # Set up times, sizes
    times = []
    sizes = []

    avg_consecutive_packets = {-1: args.num_consecutive_packets_in, 1:args.num_consecutive_packets_out}

    # note: ceil of exp is geometric
    consecutive_packets = {-1: int(math.ceil(get_exp(avg_consecutive_packets[-1], args.num_conspack_dist))),
                           1: int(math.ceil(get_exp(avg_consecutive_packets[1], args.num_conspack_dist)))}
    current_cons_packets = {-1: consecutive_packets[-1], 1:consecutive_packets[1]}
    current_is_sec = {-1: False, 1: False}
    round_robin = {-1: args.round_robin_in, 1: args.round_robin_out}

    if args.change_times > 0:
        p_loc = {-1: get_local_p(p_in, args.p_dist), 1: get_local_p(p_out, args.p_dist)}
        next_stop = get_exp(args.change_times, args.change_dist)
    elif args.change_packets > 0:
        p_loc = {-1: get_local_p(p_in, args.p_dist), 1: get_local_p(p_out, args.p_dist)}
        change_packets = get_exp(args.change_packets, args.change_dist)
    else:
        p_loc = {-1: p_in, 1: p_out}

    size_before = 0
    num_packets_sent = 0
    if args.round_robin_random:
        send_on_sec_rr = {-1: random.random() > 0.5, 1: random.random() > 0.5}
    else:
        send_on_sec_rr = {-1: True, 1: True}
    stats_sec = 0
    stats_nonsec = 0
    delay_to_add = 0
    if is_train and args.delay_sec_std_train > 0:
        delay_to_add = random.normalvariate(0, args.delay_sec_std_train)
    if not is_train and args.delay_sec_std_test > 0:
        delay_to_add = random.normalvariate(0, args.delay_sec_std_test)
    for x in f:
        num_packets_sent += 1

        ts, size = get_ts_size(x)

        if args.change_times > 0:
            assert(ts != -1)  # if length is 1, no timestamp is given
            if ts >= next_stop:
                p_loc = {-1: get_local_p(p_in, args.p_dist), 1: get_local_p(p_out, args.p_dist)}
                next_stop += get_exp(args.change_times, args.change_dist)
        elif args.change_packets > 0:
            if num_packets_sent == change_packets:
                p_loc = {-1: get_local_p(p_in, args.p_dist), 1: get_local_p(p_out, args.p_dist)}
                change_packets = int(math.ceil(get_exp(args.change_packets, args.change_dist)))
                num_packets_sent = 0

        size_before += size
        # select packets to send on the "secure" channel (they are simply removed)
        # must send consecutive_packets on same channel
        if current_cons_packets[np.sign(size)] < consecutive_packets[np.sign(size)]:
            current_cons_packets[np.sign(size)] += 1
            send_on_sec = current_is_sec[np.sign(size)]
        else:
            consecutive_packets[np.sign(size)] = int(math.ceil(get_exp(avg_consecutive_packets[np.sign(size)], args.num_conspack_dist)))
            current_cons_packets[np.sign(size)] = 1
            if original_probas[np.sign(size)] == 0.5 and round_robin[np.sign(size)]:
                # round robin: change technology (round robin only if proba is 0.5)
                send_on_sec_rr[np.sign(size)] = not send_on_sec_rr[np.sign(size)]
                send_on_sec = send_on_sec_rr[np.sign(size)]
                consecutive_packets[np.sign(size)] = avg_consecutive_packets[np.sign(size)]  # with round robin, fixed
            else:
                if random.random() <= p_loc[np.sign(size)]:
                    send_on_sec = False
                else:
                    send_on_sec = True
            current_is_sec[np.sign(size)] = send_on_sec
        if send_on_sec:
            stats_sec += abs(size)
            if (is_train and args.delay_sec_std_train > 0) or (not is_train and args.delay_sec_std_test > 0):
                # instead of removing, add delay
                ts += delay_to_add
            else:
                # remove packet (sent on secure technology)
                continue
        else:
            stats_nonsec += abs(size)

        if args.only_in and size > 0 or args.only_out and size < 0:
            # outgoing packet
            continue
        if ts >= 0:
            times.append(ts)
        sizes.append(size)

    stats_total = stats_sec + stats_nonsec

    f.close()

    size_after = sum(map(abs, sizes))

    # Extract features. All features are non-negative numbers or X.
    # try:
    name_features = []
    if type == "Wang" or type == "W":
        features = extract_Wang(times, sizes)
        # for Wang, write them in a file (learner is C++ file)
        fout = open(ffiles_loc + ffile_name, "w")
        for x in features:
            fout.write(repr(x) + " ")
        fout.close()
    elif type == "Panchenko" or type == "P":
        features = extract_Panchenko(times, sizes)
    elif type == "Hayes" or type == "H":
        features, name_features = extract_Hayes(times, sizes, args.factors_p_est, args.num_consecutive_packets_out) # n_cons_out = n_cons_in)
    else:
        print("Unknown type %s" % type)
        features = []

    sys.stdout.write("\r---- file " + fname + " ("+str(len(features))+" features) ----")
    sys.stdout.flush()

    return np.array(features), name_features, size_before, size_after


def get_local_p(p, p_dist):
    if p == 0 or p == 1:
        loc_p = p
    else:
        if p_dist == "uniform":
            s = round(min(p, 1-p),2)
            loc_p = p - s + 2*s*np.random.uniform()
        elif p_dist == "fixed":
            # fixed
            loc_p = p

    return min(1,max(0,loc_p))  # between 0 and 1


def extract_train_pool(args_loc):
    train_features = {}
    for fname in args_loc.args.train_files:
        for directory_train in args_loc.args.directories_train:

            # there can be files written as X_Y (original data) or X_Y,Z (Z-th experiment of defended data)
            # first try with X_Y,count else X_Y

            if "." in fname:
                toks = fname.split(".")
                suffix = "." + toks[-1]
            else:
                suffix = ""

            if "," not in fname:
                fname = fname.replace(suffix, "") + "," + str(args_loc.count_train) + suffix  # first try with count

            key = fname

            if not os.path.exists(directory_train + fname):
                fname = fname.split(",")[0] + suffix  # if count is not in file name

            if "." in key:
                key_split = key.split(".")
                key = "".join(key_split[:-1])

            if "," not in key:
                key = key + "," + str(args_loc.count_train)

            fout = key + suffix

            key = directory_train + key

            if args_loc.type == "W":
                key += suffix

            p_train_in_loc = get_local_p(args_loc.p_train_in, args_loc.args.p_dist)
            p_train_out_loc = get_local_p(args_loc.p_train_out, args_loc.args.p_dist)

            original_probas = {-1: args_loc.p_train_in, 1: args_loc.p_train_out}

            features_loc, name_features, size_before, size_after = write_features \
                (directory_train, fname, p_train_in_loc, p_train_out_loc, args_loc.ffiles_loc, args_loc.type, args_loc.args,
                 fout, True, original_probas)
            if len(features_loc) > 0:
                train_features[key] = features_loc

    sys.stdout.write("\nTrain files #%d done\n" % args_loc.count_exp)
    sys.stdout.flush()

    if args_loc.type == "W":  # used for computing weights with Wang
        for fname in args_loc.args.weight_files:
            for directory_train in args_loc.args.directories_train:

                if "." in fname:
                    toks = fname.split(".")
                    suffix = "." + toks[-1]
                else:
                    suffix = ""

                if "," not in fname:
                    fname = fname.replace(suffix, "") + "," + str(args_loc.count_train) + suffix  # first try with count

                key = fname

                if not os.path.exists(directory_train + fname):
                    fname = fname.split(",")[0] + suffix  # if count is not in file name

                if "." in key:
                    key_split = key.split(".")
                    key = "".join(key_split[:-1])

                if "," not in key:
                    key = key + "," + str(args_loc.count_train)

                fout = key + suffix

                key = directory_train + key

                if args_loc.type == "W":
                    key += suffix

                if key not in train_features and key not in args_loc.all_test_features:

                    p_train_in_loc = get_local_p(args_loc.p_train_in, args_loc.args.p_dist)
                    p_train_out_loc = get_local_p(args_loc.p_train_out, args_loc.args.p_dist)

                    original_probas = {-1: args_loc.p_train_in, 1: args_loc.p_train_out}

                    features_loc, name_features, size_before, size_after = write_features \
                        (directory_train, fname, p_train_in_loc, p_train_out_loc, args_loc.ffiles_loc, args_loc.type,
                         args_loc.args, fout, True, original_probas)
                    if len(features_loc) > 0:
                        train_features[key] = features_loc

        sys.stdout.write("\nWeight files #%d done\n" % args_loc.count_exp)
        sys.stdout.flush()

    return train_features


def fextractor_pool(args, ffiles_loc, type):

    t_beg = time.time()

    name_features = []
    # test files
    all_test_features = {}
    total_size_before_test = 0
    total_size_after_test = 0

    for fname in args.test_files:

        # there can be files written as X_Y (original data) or X_Y,Z (Z-th experiment of defended data)
        # first try with X_Y,0 else X_Y

        if "." in fname:
            toks = fname.split(".")
            suffix = "." + toks[-1]
        else:
            suffix = ""

        if "," not in fname:
            fname = fname.replace(suffix, "") + "," + str(args.count_test) + suffix  # first try with count

        key = fname

        if not os.path.exists(args.directory_test + fname):
            fname = fname.split(",")[0] + suffix  # if count is not in file name

        if "." in key:
            key_split = key.split(".")
            key = "".join(key_split[:-1])

        if "," not in key:
            key = key + "," + str(args.count_test)

        fout = key + suffix

        key = args.directory_test + key

        if type == "Wang" or type == "W":
            key += suffix

        p_test_in_loc = get_local_p(args.p_test_in, args.p_dist)
        p_test_out_loc = get_local_p(args.p_test_out, args.p_dist)

        original_probas = {-1: args.p_test_in, 1: args.p_test_out}

        features_loc, name_features, size_before, size_after = write_features\
            (args.directory_test, fname, p_test_in_loc, p_test_out_loc, ffiles_loc, type, args, fout, False, original_probas)
        total_size_before_test += size_before
        total_size_after_test += size_after
        if len(features_loc) > 0:
            all_test_features[key] = features_loc

    sys.stdout.write("\nTest files done\n")
    sys.stdout.flush()

    total_size_before_train = 0
    total_size_after_train = 0
    # train with train + weight
    all_train_features = {}
    count_exp = 0
    all_args_loc = []
    logger = args.logger
    args.logger = None

    count_train = args.count_train
    for (p_train_in, p_train_out) in zip(args.p_train_in, args.p_train_out):

        args_loc = types.SimpleNamespace()
        args_loc.args = args
        args_loc.p_train_out = p_train_out
        args_loc.p_train_in = p_train_in
        args_loc.ffiles_loc = ffiles_loc
        args_loc.count_exp = count_exp
        args_loc.all_test_features = all_test_features
        args_loc.type = type
        args_loc.count_train = count_train

        all_args_loc.append(args_loc)

        count_exp += 1
        count_train += 1

    args.count_train = count_train

    pool = Pool()
    results = pool.map(extract_train_pool, all_args_loc)

    for train_features in results:
        for exp in train_features:
            all_train_features[exp] = train_features[exp]

    sys.stdout.write("\nDone in "+str(time.time()-t_beg)+" seconds\n")
    sys.stdout.flush()

    args.logger = logger

    if total_size_before_test > 0:
        print("Total test size: before=%d B, after=%d B (%f sent on secure channel)" %
              (total_size_before_test, total_size_after_test, 100-float(100*total_size_after_test)/total_size_before_test))

    if total_size_before_train > 0:
        print("Total train size: before=%d B, after=%d B (%f sent on secure channel)" %
              (total_size_before_train, total_size_after_train, 100-float(100*total_size_after_train)/total_size_before_train))

    return all_test_features, all_train_features, name_features


def fextractor(args, ffiles_loc, type):

    t_beg = time.time()

    name_features = []
    # test files
    all_test_features = {}
    total_size_before_test = 0
    total_size_after_test = 0
    for fname in args.test_files:

        # there can be files written as X_Y (original data) or X_Y,Z (Z-th experiment of defended data)
        # first try with X_Y,0 else X_Y

        if "." in fname:
            toks = fname.split(".")
            suffix = "." + toks[-1]
        else:
            suffix = ""

        if "," not in fname:
            fname = fname.replace(suffix, "") + "," + str(args.count_test) + suffix  # first try with count

        key = fname

        if not os.path.exists(args.directory_test + fname):
            fname = fname.split(",")[0] + suffix  # if count is not in file name

        if "." in key:
            key_split = key.split(".")
            key = "".join(key_split[:-1])

        if "," not in key:
            key = key + "," + str(args.count_test)

        fout = key + suffix

        key = args.directory_test + key

        if type == "Wang" or type == "W":
            key += suffix

        p_test_in_loc = get_local_p(args.p_test_in, args.p_dist)
        p_test_out_loc = get_local_p(args.p_test_out, args.p_dist)

        args.original_probas = {-1: args.p_test_in, 1: args.p_test_out}

        features_loc, name_features, size_before, size_after = write_features\
            (args.directory_test, fname, p_test_in_loc, p_test_out_loc, ffiles_loc, type, args, fout, False)
        total_size_before_test += size_before
        total_size_after_test += size_after
        if len(features_loc) > 0:
            all_test_features[key] = features_loc

    sys.stdout.write("\nTest files done\n")
    sys.stdout.flush()

    total_size_before_train = 0
    total_size_after_train = 0
    # train with train + weight
    all_train_features = {}
    count_exp = 0
    for (p_train_in, p_train_out) in zip(args.p_train_in, args.p_train_out):
        for fname in args.train_files:
            for directory_train in args.directories_train:

                # there can be files written as X_Y (original data) or X_Y,Z (Z-th experiment of defended data)
                # first try with X_Y,count else X_Y

                if "." in fname:
                    toks = fname.split(".")
                    suffix = "." + toks[-1]
                else:
                    suffix = ""

                if "," not in fname:
                    fname = fname.replace(suffix, "") + "," + str(args.count_train) + suffix  # first try with count

                key = fname

                if not os.path.exists(directory_train + fname):
                    fname = fname.split(",")[0] + suffix  # if count is not in file name

                if "." in key:
                    key_split = key.split(".")
                    key = "".join(key_split[:-1])

                if "," not in key:
                    key = key + "," + str(args.count_train)

                fout = key + suffix

                key = directory_train + key

                if type == "W":
                    key += suffix

                p_train_in_loc = get_local_p(p_train_in, args.p_dist)
                p_train_out_loc = get_local_p(p_train_out, args.p_dist)

                args.original_probas = {-1: p_train_in, 1: p_train_out}

                features_loc, name_features, size_before, size_after = write_features\
                    (directory_train, fname, p_train_in_loc, p_train_out_loc, ffiles_loc, type, args, fout, True)
                total_size_before_train += size_before
                total_size_after_train += size_after
                if len(features_loc) > 0:
                    all_train_features[key] = features_loc

        sys.stdout.write("\nTrain files #%d done\n" % count_exp)
        sys.stdout.flush()

        if type == "W": # used for computing weights with Wang
            for fname in args.weight_files:
                for directory_train in args.directories_train:

                    if "." in fname:
                        toks = fname.split(".")
                        suffix = "." + toks[-1]
                    else:
                        suffix = ""

                    if "," not in fname:
                        fname = fname.replace(suffix, "") + "," + str(args.count_train) + suffix  # first try with count

                    key = fname

                    if not os.path.exists(directory_train + fname):
                        fname = fname.split(",")[0] + suffix  # if count is not in file name

                    if "." in key:
                        key_split = key.split(".")
                        key = "".join(key_split[:-1])

                    if "," not in key:
                        key = key + "," + str(args.count_train)

                    fout = key + suffix

                    key = directory_train + key

                    if type == "W":
                        key += suffix

                    if key not in all_train_features and key not in all_test_features:

                        p_train_in_loc = get_local_p(p_train_in, args.p_dist)
                        p_train_out_loc = get_local_p(p_train_out, args.p_dist)

                        args.original_probas = {-1: p_train_in, 1: p_train_out}

                        features_loc, name_features, size_before, size_after = write_features \
                            (directory_train, fname, p_train_in_loc, p_train_out_loc, ffiles_loc, type,
                             args, fout, True)
                        total_size_before_train += size_before
                        total_size_after_train += size_after
                        if len(features_loc) > 0:
                            all_train_features[key] = features_loc

            sys.stdout.write("\nWeight files #%d done\n" % count_exp)
            sys.stdout.flush()
        args.count_train += 1
        count_exp += 1

    sys.stdout.write("\nDone in "+str(time.time()-t_beg)+" seconds\n")
    sys.stdout.flush()

    if total_size_before_test > 0:
        print("Total test size: before=%d B, after=%d B (%f sent on secure channel)" %
              (total_size_before_test, total_size_after_test, 100-float(100*total_size_after_test)/total_size_before_test))

    if total_size_before_train > 0:
        print("Total train size: before=%d B, after=%d B (%f sent on secure channel)" %
              (total_size_before_train, total_size_after_train, 100-float(100*total_size_after_train)/total_size_before_train))

    return all_test_features, all_train_features, name_features

