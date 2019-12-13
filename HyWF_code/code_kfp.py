from utils import *
from sklearn.ensemble import RandomForestClassifier
import types
from multiprocessing import Pool
import itertools
import math


# three functions for Hayes openworld
def RF_openworld(tr_data, tr_label, te_data, te_label):
    # Produces leaf vectors used for classification.

    t_beg = time.time()
    sys.stdout.write("Training random forest...\n")
    sys.stdout.flush()
    model = RandomForestClassifier(n_jobs=n_jobs, n_estimators=num_Trees, oob_score=True, max_depth=70)
    model.fit(tr_data, tr_label)

    all_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
    print("Average depth: %f, max depth:%d" % (float(np.mean(all_depths)), int(max(all_depths))))

    train_leaf = zip(model.apply(tr_data), tr_label)
    test_leaf = zip(model.apply(te_data), te_label)

    del model

    sys.stdout.write("Done in " + str(time.time() - t_beg) + " seconds\n")
    sys.stdout.flush()

    return train_leaf, test_leaf


def distances_old(train_leaf, test_leaf, keep_top=100, knn=3):
    # calculate distance from test instance between each training instance (which are used as labels) and writes to file
    # Default keeps the top 100 instances closest to the instance we are testing.

    results = {}

    # Make into numpy arrays
    train_leaf = [(np.array(l, dtype=int), v) for l, v in train_leaf]
    test_leaf = [(np.array(l, dtype=int), v) for l, v in test_leaf]

    t_beg = time.time()
    sys.stdout.write("Training random forest...\n")
    sys.stdout.flush()

    for i, instance in enumerate(test_leaf):
        if i%100==0:
            #print("instance: %d" % len(instance[0]))
            sys.stdout.write("\r%d out of %d" %(i, len(test_leaf)))
            sys.stdout.flush()

        # TODO: can we make that more efficient?
        fname = '%s_%s' %(instance[1], i)
        temp = []
        for item in train_leaf:
            #if i % 100 == 0:
            #    print("item: %d" % len(item[0]))
            # vectorize the average distance computation
            d = np.sum(item[0] != instance[0]) / float(item[0].size)
            if d == 1.0:
                continue
            temp.append((d, instance[1], item[1]))
        tops = sorted(temp)[:keep_top]
        results[fname] = tops

    sys.stdout.write("\nDone in " + str(time.time() - t_beg) + " seconds\n")
    sys.stdout.flush()

    # Get knn closest predictions, predict monitored if all agree, else predict unmonitored
    guessed = {}
    for fname in results:
        internal_test = []
        for i in range(0,knn):
            predicted_label = results[fname][i][2]
            internal_test.append(predicted_label)
        if len(list(set(internal_test))) == 1:
            # all values are same: predict monitored
            guessed[fname] = str(predicted_label)
        else:
            guessed[fname] = "-1"

    return guessed


def compute_distance(args_dist):
    instance_mat = np.tile(args_dist.instance[0], (len(args_dist.train_leaf), 1))
    sum_arr = np.sum(np.not_equal(instance_mat, args_dist.train_mat) / np.shape(args_dist.train_mat)[1], axis=1)
    vals = [item[1] for item in args_dist.train_leaf]

    temp = [(x, args_dist.instance[1], v) for x, v in zip(sum_arr, vals) if x < 1.0]

    tops = sorted(temp)[:args_dist.keep_top]

    return tops


def distances(train_leaf, test_leaf, keep_top=100, knn=3):
    # calculate distance from test instance between each training instance (which are used as labels) and writes to file
    # Default keeps the top 100 instances closest to the instance we are testing.

    # optimization of the original function (distances_old)

    results = {}

    # Make into numpy arrays
    it1train, it2train = itertools.tee(train_leaf)  # create two iterators
    train_mat = [l for l, v in it1train]
    train_mat = np.array(train_mat)
    train_leaf = [(np.array(l, dtype=int), v) for l, v in it2train]
    test_leaf = [(np.array(l, dtype=int), v) for l, v in test_leaf]

    t_beg = time.time()
    sys.stdout.write("Training random forest...\n")
    sys.stdout.flush()

    #if args.single_thread:
    if True:  # multi thread does not work
        for i, instance in enumerate(test_leaf):
            sys.stdout.write("\r%d out of %d" %(i, len(test_leaf)))
            sys.stdout.flush()

            # optimization of the original function with matrices
            fname = '%s_%s' % (instance[1], i)
            instance_mat = np.tile(instance[0], (len(train_leaf),1))
            sum_arr = np.sum(np.not_equal(instance_mat, train_mat)/np.shape(train_mat)[1], axis=1)
            vals = [item[1] for item in train_leaf]

            temp = [(x,instance[1],v) for x,v in zip(sum_arr,vals) if x < 1.0]

            tops = sorted(temp)[:keep_top]
            results[fname] = tops
    else:
        all_args_dist = []
        for i, instance in enumerate(test_leaf):
            sys.stdout.write("\r%d out of %d" %(i, len(test_leaf)))
            sys.stdout.flush()
            # optimization of the original function with matrices
            fname = '%s_%s' % (instance[1], i)
            args_dist = types.SimpleNamespace()
            args_dist.train_leaf = train_leaf
            args_dist.train_mat = train_mat
            args_dist.keep_top = keep_top
            args_dist.instance = instance
            args_dist.fname = fname

            all_args_dist.append(args_dist)

            if len(all_args_dist) == n_jobs:
                sys.stdout.write("\rComputing %d jobs (has computed %d out of %d)..." % (n_jobs,i, len(test_leaf)))
                sys.stdout.flush()
                # compute
                pool = Pool()
                results_pool = pool.map(compute_distance, all_args_dist)

                for count in range(len(results_pool)):
                    results[all_args_dist[count].fname] = results_pool[count]
                all_args_dist = []

        # final jobs
        pool = Pool()
        results_pool = pool.map(compute_distance, all_args_dist)

        for count in range(len(results_pool)):
            sys.stdout.write("\rComputing %d final jobs..." % (len(results_pool)))
            sys.stdout.flush()
            results[all_args_dist[count].fname] = results_pool[count]




    sys.stdout.write("\nDone in " + str(time.time() - t_beg) + " seconds\n")
    sys.stdout.flush()

    # Get knn closest predictions, predict monitored if all agree, else predict unmonitored
    guessed = {}
    for fname in results:
        internal_test = []
        for i in range(0,knn):
            predicted_label = results[fname][i][2]
            internal_test.append(predicted_label)
        if len(list(set(internal_test))) == 1:
            # all values are same: predict monitored
            guessed[fname] = str(predicted_label)
        else:
            guessed[fname] = "-1"

    return guessed


# Features for Hayes et al. (Usenix 16)

# -1 is IN, 1 is OUT
# file format: "direction time size"


def neighborhood(iterable):
    iterator = iter(iterable)
    prev = (0)
    item = iterator.__next__()  # throws StopIteration if empty.
    for next in iterator:
        yield (prev,item,next)
        prev = item
        item = next
    yield (prev,item,None)


def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0
  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg
  return out


def get_pkt_list(trace_data, p=1):
    first_line = trace_data[0]
    first_line = first_line.split(" ")

    first_time = float(first_line[0])
    dta = []
    for line in trace_data:

        if p < 1 and random.random() > p:
            continue

        a = line
        b = a.split(" ")

        if float(b[1]) > 0:
            #dta.append(((float(b[0])- first_time), abs(int(b[2])), 1))
            dta.append(((float(b[0])- first_time), 1))
        else:
            #dta.append(((float(b[1]) - first_time), abs(int(b[2])), -1))
            dta.append(((float(b[0]) - first_time), -1))
    return dta


def In_Out(list_data):
    In = []
    Out = []
    for p in list_data:
        if p[1] == -1:
            In.append(p)
        if p[1] == 1:
            Out.append(p)
    return In, Out


# TIME FEATURES

def inter_pkt_time(list_data):
    times = [x[0] for x in list_data]
    if len(times) < 2:
        return []
    temp = []
    for elem,next_elem in zip(times, times[1:]+[times[0]]):
        temp.append(next_elem-elem)
    return temp[:-1]


def interarrival_times(list_data):
    In, Out = In_Out(list_data)
    IN = inter_pkt_time(In)
    OUT = inter_pkt_time(Out)
    TOTAL = inter_pkt_time(list_data)
    return IN, OUT, TOTAL


def get_p_est(times, t, n_cons):
    ns = []
    n = 0
    for i in range(len(times) - 1):
        if times[i + 1] - times[i] < t:
            n += 1
        else:
            ns.append(n)
            n = 0

    if len(ns) == 0:
        return 0

    avg = np.mean(ns)
    if avg <= n_cons:
        return 0
    else:
        return (avg - n_cons) / avg


def interarrival_maxminmeansd_stats(list_data, factors_p_est, n_cons):
    interstats = []
    In, Out, Total = interarrival_times(list_data)
    if In and Out:
        avg_in = sum(In) / float(len(In))
        avg_out = sum(Out) / float(len(Out))
        avg_total = sum(Total) / float(len(Total))
        interstats.append((max(In), max(Out), max(Total), avg_in, avg_out, avg_total, np.std(In), np.std(Out),
                           np.std(Total), np.percentile(In, 75), np.percentile(Out, 75), np.percentile(Total, 75)))
        for f in factors_p_est:
            interstats.append((get_p_est(In, avg_in*f, n_cons), get_p_est(Out, avg_out*f, n_cons)))
    elif Out and not In:
        avg_out = sum(Out) / float(len(Out))
        avg_total = sum(Total) / float(len(Total))
        interstats.append((0, max(Out), max(Total), 0, avg_out, avg_total, 0, np.std(Out), np.std(Total), 0,
                           np.percentile(Out, 75), np.percentile(Total, 75)))
        for f in factors_p_est:
            interstats.append(get_p_est(Out, avg_out*f, n_cons))
    elif In and not Out:
        avg_in = sum(In) / float(len(In))
        avg_total = sum(Total) / float(len(Total))
        interstats.append((max(In), 0, max(Total), avg_in, 0, avg_total, np.std(In), 0, np.std(Total),
                           np.percentile(In, 75), 0, np.percentile(Total, 75)))
        for f in factors_p_est:
            interstats.append(get_p_est(In, avg_in*f, n_cons))
    else:
        interstats = [(0,) * 15]
    return interstats


def time_percentile_stats(Total):
    In, Out = In_Out(Total)
    In1 = [x[0] for x in In]
    Out1 = [x[0] for x in Out]
    Total1 = [x[0] for x in Total]
    STATS = []
    if In1:
        STATS.append(np.percentile(In1, 25)) # return 25th percentile
        STATS.append(np.percentile(In1, 50))
        STATS.append(np.percentile(In1, 75))
        STATS.append(np.percentile(In1, 100))
    if not In1:
        STATS.extend(([0]*4))
    if Out1:
        STATS.append(np.percentile(Out1, 25)) # return 25th percentile
        STATS.append(np.percentile(Out1, 50))
        STATS.append(np.percentile(Out1, 75))
        STATS.append(np.percentile(Out1, 100))
    if not Out1:
        STATS.extend(([0]*4))
    if Total1:
        STATS.append(np.percentile(Total1, 25)) # return 25th percentile
        STATS.append(np.percentile(Total1, 50))
        STATS.append(np.percentile(Total1, 75))
        STATS.append(np.percentile(Total1, 100))
    if not Total1:
        STATS.extend(([0]*4))
    return STATS


def number_pkt_stats(Total):
    In, Out = In_Out(Total)
    return len(In), len(Out), len(Total)


def first_and_last_30_pkts_stats(Total):
    first30 = Total[:30]
    last30 = Total[-30:]
    first30in = []
    first30out = []
    for p in first30:
        if p[1] == -1:
            first30in.append(p)
        if p[1] == 1:
            first30out.append(p)
    last30in = []
    last30out = []
    for p in last30:
        if p[1] == -1:
            last30in.append(p)
        if p[1] == 1:
            last30out.append(p)
    stats= []
    stats.append(len(first30in))
    stats.append(len(first30out))
    stats.append(len(last30in))
    stats.append(len(last30out))
    return stats


# concentration of outgoing packets in chunks of 20 packets
def pkt_concentration_stats(Total):
    chunks= [Total[x:x+20] for x in np.arange(0, len(Total), 20)]
    concentrations = []
    for item in chunks:
        c = 0
        for p in item:
            if p[1] == 1:
                c+=1
        concentrations.append(c)
    if len(concentrations) > 0:
        return np.std(concentrations), sum(concentrations)/float(len(concentrations)), np.percentile(concentrations, 50), min(concentrations), max(concentrations), concentrations
    return 0, 0, 0, 0, 0, [0]

# Average number packets sent and received per second
def number_per_sec(Total):
    if len(Total) == 0:
        return 0, 0, 0, 0, 0, [0]
    last_time = Total[-1][0]
    first_second = math.floor(Total[0][0])
    last_second = math.ceil(last_time) - first_second
    temp = [0]*int(last_second)
    Total_cpy = list(Total)
    # modified version compared to original for efficiency
    for i in range(1, int(last_second)+1):
        if i == 1:
            c = 0
        else:
            c = temp[i-2]
        j = 0
        for p in Total_cpy:
            if p[0] - first_second <= i:
                c+=1
            else:
                Total_cpy = Total_cpy[j:]
                break
            j += 1
        temp[i-1] = c

    l = []
    for prev,item,next in neighborhood(temp):
        x = item - prev
        l.append(x)

    if len(l) == 0:
        return 0, 0, 0, 0, 0, [0]

    avg_number_per_sec = sum(l)/float(len(l))
    return avg_number_per_sec, np.std(l), np.percentile(l, 50), min(l), max(l), l


# Variant of packet ordering features from http://cacr.uwaterloo.ca/techreports/2014/cacr2014-05.pdf
def avg_pkt_ordering_stats(Total):
    c1 = 0
    c2 = 0
    temp1 = []
    temp2 = []
    for p in Total:
        if p[1] == 1:
            temp1.append(c1)
        c1+=1
        if p[1] == -1:
            temp2.append(c2)
        c2+=1

    avg_in = 0
    avg_out = 0
    std_in = 0
    std_out = 0
    if len(temp1) > 0:
        avg_in = sum(temp1)/float(len(temp1))
        std_in = np.std(temp1)
    if len(temp2) > 0:
        avg_out = sum(temp2)/float(len(temp2))
        std_out = np.std(temp2)

    return avg_in, avg_out, std_in, std_out


def perc_inc_out(Total):
    if len(Total) == 0:
        return 0, 0
    In, Out = In_Out(Total)
    percentage_in = len(In)/float(len(Total))
    percentage_out = len(Out)/float(len(Total))
    return percentage_in, percentage_out


# FEATURE FUNCTION
def extend_all_features(vals, count, all_features):
    if count >= len(all_features):
        return len(all_features)
    if not isinstance(vals,list):
        vals = [vals]
    if len(vals) == 0:
        return count
    max_ind = count+len(vals)
    if max_ind > len(all_features):
        max_ind = len(all_features)
    all_features[count:max_ind] = vals[:max_ind-count]
    count += len(vals)
    return min(count, len(all_features))


# If size information available add them in to function below
def extract_Hayes(times, sizes, factors_p_est, n_cons, max_size=175):
    list_data = list(zip(times, sizes))

    all_features = np.array([0]*max_size)
    name_features = np.array(["this is a string"]*max_size)
    count = 0
    count_names = 0

    i = 0
    # ------TIME--------
    intertimestats = [x for x in interarrival_maxminmeansd_stats(list_data, factors_p_est, n_cons)[0]]
    timestats = time_percentile_stats(list_data)
    number_pkts = list(number_pkt_stats(list_data))
    thirtypkts = first_and_last_30_pkts_stats(list_data)
    stdconc, avgconc, medconc, minconc, maxconc, conc = pkt_concentration_stats(list_data)
    #t_beg = time.time()
    #print("number_per_sec")
    avg_per_sec, std_per_sec, med_per_sec, min_per_sec, max_per_sec, per_sec = number_per_sec(list_data)
    #print(str(i)+": nsec Done in "+str(time.time() - t_beg)+" seconds")
    avg_order_in, avg_order_out, std_order_in, std_order_out = avg_pkt_ordering_stats(list_data)
    perc_in, perc_out = perc_inc_out(list_data)

    altconc = [sum(x) for x in chunkIt(conc, 70)]
    alt_per_sec = [sum(x) for x in chunkIt(per_sec, 20)]
    if len(altconc) == 70:
        altconc.append(0)
    if len(alt_per_sec) == 20:
        alt_per_sec.append(0)

    sum_intertimestats = sum(intertimestats)

    # TIME Features
    count = extend_all_features(intertimestats, count, all_features)
    count_names = extend_all_features(['intertimestats']*len(intertimestats), count_names, name_features)
    count = extend_all_features(timestats, count, all_features)
    count_names = extend_all_features(['timestats']*len(timestats), count_names, name_features)
    count = extend_all_features(number_pkts, count, all_features)
    count_names = extend_all_features(['number_pkts']*len(number_pkts), count_names, name_features)
    count = extend_all_features(thirtypkts, count, all_features)
    count_names = extend_all_features(['thirtypkts']*len(thirtypkts), count_names, name_features)
    count = extend_all_features(stdconc, count, all_features)
    count_names = extend_all_features(['stdconc'], count_names, name_features)
    count = extend_all_features(avgconc, count, all_features)
    count_names = extend_all_features(['avgconc'], count_names, name_features)
    count = extend_all_features(avg_per_sec, count, all_features)
    count_names = extend_all_features(['avg_per_sec'], count_names, name_features)
    count = extend_all_features(std_per_sec, count, all_features)
    count_names = extend_all_features(['std_per_sec'], count_names, name_features)
    count = extend_all_features(avg_order_in, count, all_features)
    count_names = extend_all_features(['avg_order_in'], count_names, name_features)
    count = extend_all_features(avg_order_out, count, all_features)
    count_names = extend_all_features(['avg_order_out'], count_names, name_features)
    count = extend_all_features(std_order_in, count, all_features)
    count_names = extend_all_features(['std_order_in'], count_names, name_features)
    count = extend_all_features(std_order_out, count, all_features)
    count_names = extend_all_features(['std_order_out'], count_names, name_features)
    count = extend_all_features(medconc, count, all_features)
    count_names = extend_all_features(['medconc'], count_names, name_features)
    count = extend_all_features(med_per_sec, count, all_features)
    count_names = extend_all_features(['med_per_sec'], count_names, name_features)
    count = extend_all_features(min_per_sec, count, all_features)
    count_names = extend_all_features(['min_per_sec'], count_names, name_features)
    count = extend_all_features(max_per_sec, count, all_features)
    count_names = extend_all_features(['max_per_sec'], count_names, name_features)
    count = extend_all_features(maxconc, count, all_features)
    count_names = extend_all_features(['maxconc'], count_names, name_features)
    count = extend_all_features(perc_in, count, all_features)
    count_names = extend_all_features(['perc_in'], count_names, name_features)
    count = extend_all_features(perc_out, count, all_features)
    count_names = extend_all_features(['perc_out'], count_names, name_features)
    count = extend_all_features(altconc, count, all_features)
    count_names = extend_all_features(['altconc']*len(altconc), count_names, name_features)
    count = extend_all_features(alt_per_sec, count, all_features)
    count_names = extend_all_features(['alt_per_sec']*len(alt_per_sec), count_names, name_features)
    count = extend_all_features(sum(altconc), count, all_features)
    count_names = extend_all_features(['altconc']*len(altconc), count_names, name_features)
    count = extend_all_features(sum(alt_per_sec), count, all_features)
    count_names = extend_all_features(['alt_per_sec']*len(alt_per_sec), count_names, name_features)
    count = extend_all_features([sum(intertimestats)], count, all_features)
    count_names = extend_all_features('sum_intertimestats', count_names, name_features)
    count = extend_all_features([sum(timestats)], count, all_features)
    count_names = extend_all_features('sum_timestats', count_names, name_features)
    count = extend_all_features([sum(number_pkts)], count, all_features)
    count_names = extend_all_features('sum_number_pkts', count_names, name_features)

    # This is optional, since all other features are of equal size this gives the first n features
    # of this particular feature subset, some may be padded with 0's if too short.

    count = extend_all_features(conc, count, all_features)
    count_names = extend_all_features(['conc'] * len(conc), count_names, name_features)
    extend_all_features(per_sec, count, all_features)
    extend_all_features(['per_sec'] * len(per_sec), count_names, name_features)

    assert len(all_features) == len(name_features)

    return all_features, name_features