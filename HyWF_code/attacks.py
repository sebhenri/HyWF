#!/usr/bin/python

from subprocess import call
import sys
import time
from gen_list import gen_list
from fextractor import fextractor, fextractor_pool
import argparse
from compute_accuracy import compute_accuracy
from sklearn import svm
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import hashlib
import logging
from utils import *
import warnings
from code_kfp import RF_openworld, distances
from code_cumul import score_accuracy_multi_class,score_accuracy_two_class

# This file contains the implementation of the attacks
# The extraction of the features is made by fextractor.py (fextractor_pool is parallelized)

warnings.filterwarnings("ignore", "numpy.dtype size changed*")
warnings.filterwarnings("ignore", "numpy.ufunc size changed*")
warnings.filterwarnings("ignore", "Solver terminated early.*")


def remove_ffiles(ffiles_loc):
    ffiles = glob.glob(ffiles_loc + "*f")
    if len(ffiles) > 0:
        print("Removing " + str(len(ffiles)) + " ffiles")
        for fname in ffiles:
            os.remove(fname)


def name_to_ind(fname):
    remove_dir = fname.split("/")[-1]
    if fname[0:2] == "-1":
        return "-1"
    return re.split("[_-]", remove_dir)[0] if "-" in remove_dir or "_" in remove_dir else "-1"


def name_to_exp(fname):
    remove_dir = fname.split("/")[-1]
    return remove_dir.split(".")[0].split(",")[0].replace("-", "_")


def conditional_str(s, b):
    return s if b else ""


def get_features(args, file_flearner_name, type):
    if args.type == "W" and args.remove_ffiles:
        remove_ffiles(args.cellf_loc)

    print("Extracting...")

    pickle_file = file_flearner_name.replace(".results", ".pickle")

    if type == "W":
        print(args.first_ffile)
        get_pickled = False
        if args.first_ffile is not None:
            if os.path.exists(pickle_file) and os.path.getmtime(pickle_file) > os.path.getmtime(args.first_ffile):
                get_pickled = True
    else:
        get_pickled = False
        if not args.force and os.path.exists(pickle_file):
            if os.path.exists(pickle_file) and os.path.getmtime(pickle_file) > os.path.getmtime(args.first_file):
                get_pickled = True

    if args.force:
        get_pickled = False

    if get_pickled:
        pickle_res = pickle.load(open(pickle_file, "rb"))
        if "all_test_features" in pickle_res and "all_train_features" in pickle_res and \
                "name_features" in pickle_res:
            print("Loading features...")
            all_test_features = pickle_res["all_test_features"]
            all_train_features = pickle_res["all_train_features"]
            name_features = pickle_res["name_features"]
            args.count_train += len(args.p_train_in)
        else:
            get_pickled = False

    if not get_pickled:
        if args.single_thread:
            all_test_features, all_train_features, name_features = fextractor(args, args.cellf_loc, type)
        else:
            all_test_features, all_train_features, name_features = fextractor_pool(args, args.cellf_loc, type)
        pickle_res = {}
        pickle_res["all_test_features"] = all_test_features
        pickle_res["all_train_features"] = all_train_features
        pickle_res["name_features"] = name_features
        pickle.dump(pickle_res, open(pickle_file, "wb"))
        print("Saving features in pickle")

    print("Learning model with %d train data, %d test data" % (len(all_train_features), len(all_test_features)))

    return all_train_features, all_test_features, name_features


def SVM_scale(x, transpose=True):
    # scale data in [-1,1]
    x_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    if transpose:
        return x_scaler.fit_transform(x.transpose()).transpose()
    return x_scaler.fit_transform(x)


def attacks(args):

    local_directory = os.path.dirname(os.path.abspath(__file__))+"/"

    # different hash for different consecutive packets (was computed without when n=20, keep it to avoid hash change)
    str_nc = conditional_str(str(args.num_consecutive_packets_in), args.num_consecutive_packets_in != 20)
    str_nc += conditional_str(args.num_conspack_dist, args.num_conspack_dist != "fixed")
    str_nc += conditional_str(str(args.num_consecutive_packets_out), args.num_consecutive_packets_out != 20)
    # different hash for different arguments
    str_nc += conditional_str(args.change_dist, args.num_conspack_dist != "fixed") # this is a mistake, but keep it for consistency
    str_nc += conditional_str(args.change_dist, args.change_dist != "fixed")
    str_nc += conditional_str(str(args.change_packets), args.change_packets > 0 )
    str_nc += conditional_str(str(args.change_times), args.change_times > 0)
    str_nc += conditional_str("rro", args.round_robin_out)
    str_nc += conditional_str("rri", args.round_robin_in)
    str_nc += conditional_str("rrr", args.round_robin_random)
    str_nc += conditional_str("nf"+str(args.test_prop), args.test_prop != 1/3)
    str_nc += conditional_str("det", args.deterministic_split)
    str_nc += "-".join(map(str, args.factors_p_est))
    if args.num_open_insts > 0 and args.num_open_insts != 9000:
        str_nc += str(args.num_open_insts)
    if args.delay_sec_std_train > 0:
        str_nc += str(args.delay_sec_std_train)
    if args.delay_sec_std_test > 0:
        str_nc += str(args.delay_sec_std_test)

    all_files_str = " ".join(args.train_directories)+args.test_directory+args.p_train_str+str_nc
    all_files_str = all_files_str.replace("//","/")
    # different file systems, replace one with the other so that it produces same hash
    if not args.dont_replace_fs:
        all_files_str = all_files_str.replace("home/90days", "dfs/ephemeral/storage")
        all_files_str = all_files_str.replace("/root", "/dfs/ephemeral/storage/shenri")

    print("String to hash is "+all_files_str)

    hash_data = hashlib.md5(all_files_str.encode()).hexdigest()
    hash_data = str(hash_data)[0:10]

    ffiles_loc = args.output_directory+"ffiles"+hash_data+"/"
    args.cellf_loc = ffiles_loc

    if not args.light:

        if (args.type == "W" or args.type == "E") and (args.compile or not os.path.exists(local_directory+"flearner")):
            print("compile flearner...")
            cmd = "g++ -o "+local_directory+"flearner "+local_directory+"flearner.cpp"
            outcode = call(cmd, shell=True)
            if outcode == 1:
                # there has been an error
                print("Error while compiling flearner")
                return

        if not os.path.exists(args.cellf_loc):
            os.makedirs(args.cellf_loc)

    open_world = args.num_open_insts > 0

    if hash_data != "":
        hash_data = "_"+hash_data
    name_results = "flearner"+hash_data+"_"+args.type[0]+("n"+args.no_type if args.no_type != "" else "")+\
                   ("" if not open_world else "m" if (args.type == "P" and args.multi_class) else "t")+\
                   ("o" if open_world else "c")+args.p_dist[0]
    if args.only_in:
        name_results += "_oi"
    if args.only_out:
        name_results += "_oo"
    if args.round_robin_in:
        name_results += "_rri"
    if args.round_robin_out:
        name_results += "_rro"
    name_results += args.affix

    all_tprs = []
    all_fprs = []
    duration = []

    args.count_train = 0

    for num_exp in range(args.first_index, args.first_index+args.num_exps):

        (args.directories_train, args.directory_test, test_files, train_files, weight_files) = gen_list(args)

        # data for fextractor
        args.test_files = test_files
        args.train_files = train_files
        args.weight_files = weight_files

        if not args.light:
            if len(weight_files) == 0:
                print("Empty files, return")
                return [], [], []

        print("Num consecutive packets: in %d, out %d (%s)"
              % (args.num_consecutive_packets_in, args.num_consecutive_packets_out, args.num_conspack_dist))
        print("Num change: times %d, packets %d (%s)"
              % (args.change_times, args.change_packets, args.change_dist))

        args.count_test = num_exp

        file_done = False
        args.file_flearner_name = args.output_directory + name_results+"_"+str(num_exp)+".results"
        print("Result file is " + args.file_flearner_name)
        args.first_file = None
        for f in os.scandir(args.train_directories[0]):
            args.first_file = args.train_directories[0]+f.name
            if not "pickle" in args.first_file: # pickle files are just metadata files
                break
        if args.first_file is None:
            print("First file is None")
            continue

        args.first_ffile = None
        if args.type == "W" or args.type == "E":
            for f in os.scandir(args.cellf_loc):
                args.first_ffile = args.cellf_loc+f.name
                break
            print("First ffile is %s" % args.first_ffile)

        if os.path.exists(args.file_flearner_name):
            # check whether results already exist
            t_first = os.path.getmtime(args.first_file)
            t_flearner = os.path.getmtime(args.file_flearner_name)
            if t_flearner > t_first:
                f = open(args.file_flearner_name, "r")
                lines = f.readlines()
                feature_ok = True
                num_line_features = 0
                if args.feature_stats:
                    feature_ok = False
                    for line in lines:
                        if "feature" in line:
                            feature_ok = True
                            num_line_features += 1
                if (len(lines) - num_line_features) > 1 and feature_ok and not args.force:
                    print("Loading results from %s" % args.file_flearner_name)
                    file_done = True
                    if args.feature_stats:
                        for line in lines:
                            if "feature" in line:
                                sys.stdout.write(line)
                else:
                    print("WARNING: Something went wrong: %d lines, %d features (feature OK %s)" %
                          (len(lines), num_line_features, str(feature_ok)))
            else:
                print("File %s was modified at %s, result file %s at %s" %
                      (args.first_file, str(t_first), args.file_flearner_name, str(t_flearner)))

        if args.read_only and not file_done:
            print("No result for this file, continue")
            continue

        if not args.light and (args.force or args.force_train or not file_done):

            # if they exist, remove files
            if os.path.exists(args.output_directory + name_results + ".log"):
                os.remove(args.output_directory + name_results + ".log")
            if os.path.exists(args.file_flearner_name):
                os.remove(args.file_flearner_name)

            t_beg_all_exp = time.time()
            file_flearner = open(args.file_flearner_name, "a")
            print_line = str(t_beg_all_exp)+": p_trains are in: "+("-".join(map(str,args.p_train_in)))\
                         + ", out: "+("-".join(map(str,args.p_train_out)))+"p_tests are in:"+str(args.p_test_in)\
                         + ", out:"+str(args.p_test_out)+", data is in train:"+(", ".join(args.train_directories))\
                         + ", test:"+args.test_directory
            file_flearner.write(print_line+"\n")
            file_flearner.close()
            print(print_line)

            if args.type == "W" or args.type == "E":
                # k-NN attack (uses flearner.cpp)

                if args.type == "E":
                    file_flearner_name = args.file_flearner_name.replace(hash_data + "_E", hash_data + "_EW")
                else:
                    file_flearner_name = args.file_flearner_name
                file_flearner_name = file_flearner_name+".tmp"
                all_train_features, all_test_features, name_features = get_features(args, file_flearner_name, "W")

                if os.path.exists(file_flearner_name):
                    os.remove(file_flearner_name)

                dict_guessed_values = {}
                dict_max_diff = {}
                # write local option name for C++ file flearner
                optfname_c = args.output_directory + "options" + args.affix + "-c"
                f = open(optfname_c, "w")
                f.write("OUTPUT_LOC\t"+args.output_directory+"\n")
                f.write("RESULT_FILE\t"+file_flearner_name+"\n")
                f.write("CLOSED_SITENUM\t"+str(args.num_sites)+"\n")
                f.write("OPEN_INSTNUM\t"+str(args.num_open_insts)+"\n")
                f.write("CELLF_LOC\t"+args.cellf_loc+"\n")
                f.write("TRAIN_LIST\t"+args.output_directory+"trainlist"+"\n")
                f.write("TEST_LIST\t"+args.output_directory+"testlist"+"\n")
                # write train and test files
                first_file = ""
                f_train = open(args.output_directory+"trainlist", "w")
                feat_num = 0
                for k in all_train_features:
                    ffile_name = args.cellf_loc+k.split("/")[-1]
                    f_train.write(ffile_name+"\n")
                    if first_file == "":
                        first_file = k.split("/")[-1]
                    feat_num = len(all_train_features[k])
                f.write("FEAT_NUM\t" + str(feat_num) + "\n")
                f_train.close()
                del all_train_features
                f_test = open(args.output_directory+"testlist", "w")
                for k in all_test_features:
                    ffile_name = args.cellf_loc+k.split("/")[-1]
                    f_test.write(ffile_name+"\n")
                f_test.close()
                if args.num_neighbors is not None:
                    f.write("NUM_NEIGHBORS\t" + str(args.num_neighbors)+"\n")
                if "." in first_file:
                    first_file = first_file+"f"
                else:
                    first_file = first_file+".f"
                f.write("FIRST_FILE\t"+first_file+"\n")
                f.close()

                print("flearner...")
                cmd = local_directory + "flearner " + optfname_c

                t_beg = time.time()
                call(cmd, shell=True)
                sys.stdout.write("Done in " + str(time.time() - t_beg) + " seconds\n")
                sys.stdout.flush()
                t_end_fit = time.time()

                file_flearner = open(file_flearner_name, "r")
                lines = file_flearner.readlines()
                file_flearner.close()

                for line in lines:
                    if "p_train" in line or "feature" in line:
                        continue
                    if "done" in line:
                        done_line = line
                        continue
                    tokens = line.strip().split()
                    exp = tokens[0]
                    toks = re.split("_", exp)
                    if len(toks) > 0 and int(toks[0]) >= 0:  # foreground page
                        if open_world and not args.multi_class:
                            true_site = 0
                        else:
                            true_site = int(toks[0])
                    else:  # background page
                        true_site = -1
                    values = list(map(float, tokens[1:]))
                    sum_values = sum(values)
                    if sum_values == 0:
                        print("Error with exp %s, continue" % exp)
                        continue
                    values = list(map(lambda x:x/sum_values, values))  # sum to 1 like a proba
                    sorted_values = sorted(values, reverse=True)
                    guessed_site = np.argmax(values)
                    if open_world:
                        if guessed_site == len(values) - 1:
                            # guessed white list
                            guessed_site = -1
                        elif not args.multi_class:
                            # guessed black list
                            guessed_site = 0
                    #print(exp)
                    dict_guessed_values[exp] = guessed_site
                    dict_max_diff[exp] = (sorted_values[0] - sorted_values[1])

                guessed_values_W = dict(dict_guessed_values)
                max_diff_W = dict(dict_max_diff)

            max_diff_P = {}
            if (args.type == "P" or args.type == "E") and not args.no_type == "P":
                # CUMUL attack
                if args.type == "E":
                    file_flearner_name = args.file_flearner_name.replace(hash_data + "_E", hash_data + "_EP")
                else:
                    file_flearner_name = args.file_flearner_name
                all_train_features, all_test_features, name_features = get_features(args, file_flearner_name, "P")

                dict_guessed_values = {}
                dict_max_diff = {}
                X_test = np.array([v for (k, v) in all_test_features.items()]).astype(float)
                X_train = np.array([v for (k, v) in all_train_features.items()]).astype(float)
                y_train = np.array([name_to_ind(fname) for fname in all_train_features.keys()])

                del all_train_features

                # scale in [-1,1]
                #X_train = SVM_scale(X_train)
                #X_test = SVM_scale(X_test)

                # Panchenko et al. scale in [-1,1], i.e. MaxMinScaler (in SVM_scale), but StandardScaler gives better results
                x_scaler = preprocessing.StandardScaler()
                X_train = x_scaler.fit_transform(X_train)
                X_test = x_scaler.transform(X_test)

                print(X_test)
                true_values = [name_to_ind(fname) for fname in
                               all_test_features.keys()]
                exps = [name_to_exp(fname)for fname in all_test_features.keys()]

                if args.multi_class or not open_world:
                    score_fun = metrics.make_scorer(score_accuracy_multi_class, greater_is_better=True)
                else:
                    score_fun = metrics.make_scorer(score_accuracy_two_class, greater_is_better=True)
                # parameters given by Panchenko et al.
                tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2e-3, 2e-1, 2e1, 2e3], 'C': [2e11, 2e13, 2e15, 2e17]}]
                clf = model_selection.GridSearchCV(svm.SVC(verbose=0, probability=True),  tuned_parameters,
                                                   scoring=score_fun, verbose=1, n_jobs=n_jobs, cv=5)

                t_beg = time.time()
                sys.stdout.write("Learning SVM model with "+str(X_train.shape[0])+" train values, "+str(X_train.shape[1])+" features\n")
                sys.stdout.flush()
                clf.fit(X_train, y_train)
                sys.stdout.write("Done in "+str(time.time() - t_beg)+" seconds\n")
                sys.stdout.flush()
                if hasattr(clf, 'best_params_'):
                    # GridSearchCV
                    print("Best parameters set found on development set: C="\
                          +("%.3g" % clf.best_params_['C'])+", gamma="+("%.3g" % clf.best_params_['gamma']))
                t_end_fit = time.time()
                guessed_values = np.array(clf.predict(X_test))
                predict_probas = np.array(clf.predict_proba(X_test))
                sorted_probas = [sorted(x, reverse=True) for x in predict_probas]
                max_diff = [(x[0] - x[1]) for x in sorted_probas]

                for count in range(len(exps)):
                    exp = exps[count]
                    #print(exp)
                    guessed_site = guessed_values[count]
                    if int(name_to_ind(exp)) != int(true_values[count]):
                        print("Values do not correspond for " + exp + " (" + str(true_values[count]) + ")")
                    dict_guessed_values[exp] = guessed_site
                    dict_max_diff[exp] = max_diff[count]

                # free variables
                del clf
                del X_train
                del X_test
                del y_train

                guessed_values_P = dict(dict_guessed_values)
                max_diff_P = dict(dict_max_diff)

            if args.type == "H" or args.type == "E":
                # k-fingerprinting attack
                if args.type == "E":
                    file_flearner_name = args.file_flearner_name.replace(hash_data + "_E", hash_data + "_EH")
                else:
                    file_flearner_name = args.file_flearner_name
                all_train_features, all_test_features, name_features = get_features(args, file_flearner_name, "H")

                dict_guessed_values = {}
                dict_max_diff = {}
                X_train = np.array([v for (k, v) in all_train_features.items()]).astype(float)
                y_train = np.array([name_to_ind(fname) for fname in all_train_features.keys()])

                del all_train_features

                X_test = np.array([v for (k, v) in all_test_features.items()]).astype(float)
                X_scaler = preprocessing.StandardScaler()
                X_train = X_scaler.fit_transform(X_train)
                X_test = X_scaler.transform(X_test)
                true_values = [name_to_ind(fname) for fname in
                               all_test_features.keys()]
                exps = [name_to_exp(fname)for fname in all_test_features.keys()]

                del X_scaler

                if open_world:
                    t_beg = time.time()

                    train_leaf, test_leaf = RF_openworld(X_train, y_train, X_test, true_values)

                    print("Model learnt in %s..." % repr(time.time() - t_beg))

                    del X_train
                    del y_train
                    del X_test

                    t_beg = time.time()
                    guessed = distances(train_leaf, test_leaf)

                    print("Computed distances in %s..." % repr(time.time() - t_beg))

                    guessed = [(re.split("_", fname)[0], label) for fname,label in guessed.items()]
                    t_end_fit = time.time()
                    true_values, guessed_values = zip(*guessed)

                else:
                    t_beg = time.time()
                    sys.stdout.write("Training random forest...\n")
                    sys.stdout.flush()
                    model = RandomForestClassifier(n_estimators=num_Trees, oob_score=True, n_jobs=n_jobs, max_depth=70)
                    model.fit(X_train, y_train)
                    t_end_fit = time.time()
                    sys.stdout.write("Model fit done...\n")
                    sys.stdout.flush()

                    all_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
                    print("Average depth: %f, max depth:%d" % (float(np.mean(all_depths)), int(max(all_depths))))

                    if args.feature_stats:
                        importances = model.feature_importances_
                        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                                     axis=0)
                        indices = np.argsort(importances)[::-1]

                        # Print the feature ranking
                        print("Feature ranking (out of %d features):" % len(importances))

                        file_flearner = open(file_flearner_name, "a")
                        for f in range(X_train.shape[1]):
                            ln = "%d. feature %s (%d): %f, std=%f" % (f + 1, name_features[indices[f]], indices[f],
                                                                      importances[indices[f]], std[indices[f]])
                            print(ln)
                            file_flearner.write(ln+"\n")

                        file_flearner.close()

                    guessed_values = np.array(model.predict(X_test))
                    predict_probas = np.array(model.predict_proba(X_test))
                    sorted_probas = [sorted(x, reverse=True) for x in predict_probas]
                    max_diff = [(x[0] - x[1]) for x in sorted_probas]

                    sys.stdout.write("Done in "+str(time.time() - t_beg)+" seconds\n")
                    sys.stdout.flush()

                    for count in range(len(exps)):
                        exp = exps[count]
                        #print(exp)
                        guessed_site = guessed_values[count]
                        if int(name_to_ind(exp)) != int(true_values[count]):
                            print("Values do not correspond for " + exp + " (" + str(true_values[count]) + ")")
                        dict_guessed_values[exp] = guessed_site
                        dict_max_diff[exp] = max_diff[count]

                    # free variables
                    del model
                    del X_train
                    del X_test
                    del y_train

                    guessed_values_H = dict(dict_guessed_values)
                    max_diff_H = dict(dict_max_diff)

            if open_world and args.type == "H":
                # TODO: consistent method
                # write results in flearner.results for consistency between methods
                file_flearner = open(file_flearner_name, "a")
                num_classes = open_world + (1 + max(list(map(int, true_values))))
                for count in range(len(guessed_values)):
                    values = ['0'] * num_classes
                    id_val = int(guessed_values[count])
                    if id_val == -1:
                        id_val = len(values) - 1
                    values[id_val] = '1'
                    values_str = "\t".join(values)
                    file_flearner.write(true_values[count] + "_" + str(count) + "\t" + values_str + "\n")
                now = time.time()
                total_time = t_end_fit-t_beg_all_exp
                file_flearner.write(str(now) + ": Experiment done in "+str(total_time)+" seconds\n")
                file_flearner.close()
                print("Saved all results in %s." % file_flearner_name)
            else:
                if len(dict_guessed_values) == 0:
                    print("Could not perform the attack")
                else:
                    # write results in flearner.results for compatibility with older version and consistency between methods
                    # TODO: directly compute the accuracy here without writing in a file
                    file_flearner = open(args.file_flearner_name, "a")
                    count = 0
                    true_values = [name_to_ind(exp) for exp in dict_guessed_values]
                    num_classes = open_world + (1 + max(list(map(int, true_values))))
                    print("Has %d experiments" % len(dict_guessed_values))
                    for exp in dict_guessed_values:
                        true_value = name_to_ind(exp)
                        if "-" in exp or "_" in exp:
                            indx = re.split("[_-]", re.split(",",re.split("/",exp)[-1])[0])[-1]
                        else:
                            indx = exp
                        values = ['0'] * num_classes
                        if args.type == "E":
                            stop = False
                            if exp not in max_diff_W:
                                print("WARNING: %s not in max_diff_W" % exp)
                                stop = True
                            if exp not in max_diff_P:
                                print("WARNING: %s not in max_diff_P" % exp)
                                stop = True
                            if exp not in max_diff_H:
                                print("WARNING: %s not in max_diff_H" % exp)
                                stop = True
                            if exp not in guessed_values_W:
                                print("WARNING: %s not in guessed_values_W" % exp)
                                stop = True
                            if exp not in guessed_values_P:
                                print("WARNING: %s not in guessed_values_P" % exp)
                                stop = True
                            if exp not in guessed_values_H:
                                print("WARNING: %s not in guessed_values_H" % exp)
                                stop = True
                            if stop:
                                continue
                            max_diffs = [max_diff_W[exp], max_diff_P[exp], max_diff_H[exp]]
                            best_attack = np.argmax(max_diffs)
                            if best_attack == 0:
                                id_val = int(guessed_values_W[exp])
                            elif best_attack == 1:
                                id_val = int(guessed_values_P[exp])
                            elif best_attack == 2:
                                id_val = int(guessed_values_H[exp])
                        else:
                            id_val = int(dict_guessed_values[exp])
                        if id_val == -1:
                            id_val = len(values) - 1
                        values[id_val] = '1'
                        values_str = "\t".join(values)
                        file_flearner.write(true_value + "_" + indx + "\t" + values_str + "\n")
                        count += 1
                    now = time.time()
                    total_time = t_end_fit-t_beg_all_exp
                    file_flearner.write(str(now) + ": Experiment done in "+str(total_time)+" seconds\n")
                    file_flearner.close()
                    print("Saved all results in %s." % args.file_flearner_name)

            del all_test_features
            del name_features

        else:
            args.count_train += len(args.p_train_in)

        try:
            tpr, fpr, dur = compute_accuracy([args.file_flearner_name], open_world, twoclass=(not args.multi_class),
                                        verbose=args.verbose <= INFO)
            if dur != -1:
                duration.append(dur)
        except Exception as e:
            print("Error in compute_accuracy")
            # remove ffiles in case of error before raising error
            if args.type == "W" and args.remove_ffiles:
                remove_ffiles(args.cellf_loc)
            raise e

        if args.dont_save:
            os.remove(args.file_flearner_name)

        if args.type == "W" and args.remove_ffiles:
            remove_ffiles(args.cellf_loc)

        if tpr > 0:
            all_tprs.append(tpr)
        if fpr > 0:
            all_fprs.append(fpr)

    if len(all_tprs) > 0:
        avg_tpr = float(np.mean(all_tprs))
        std_tpr = float(np.std(all_tprs))
    else:
        avg_tpr = -1
        std_tpr = -1
    if len(all_fprs) > 0:
        avg_fpr = float(np.mean(all_fprs))
        std_fpr = float(np.std(all_fprs))
    else:
        avg_fpr = -1
        std_fpr = -1
    print("Average TPR is %f, std is %f (TPRs are %s)" % (avg_tpr, std_tpr,
                                                          " ".join(map(lambda x:("%.4f" % x),all_tprs))))
    if open_world:
        print("Average FPR is %f, std is %f (FPRs are %s)" % (avg_fpr, std_fpr,
                                                              " ".join(map(lambda x:("%.4f" % x),all_fprs))))

    print("Attack done\n\n")

    return all_tprs, all_fprs, duration


def parse_p_trains(p_trains_str):
    p_trains = []
    for s in p_trains_str:
        if "*" in s:
            # X*p
            toks = s.split("*")
            n = int(toks[0])
            p = float(toks[1])
            p_trains.extend(n*[p])
        else:
            p_trains.append(float(s))

    return p_trains


def prob_to_zfill_str(p):
    if type(p) is float or type(p) is int:
        return str(int(p * 100)).zfill(2)
    if "*" in p:
        # X*p
        toks = p.split("*")
        assert(len(toks) == 2)
        return toks[0]+"*"+str(int(float(toks[1]) * 100)).zfill(2)
    return str(int(float(p) * 100)).zfill(2)


def get_args(argv=None, light=False):  # if light, only returns minimal args with no action
    parser = argparse.ArgumentParser()

    parser.add_argument('-tr', '--train_directories', nargs='+', default=[],
                        help="List of directories with train data (if not set, use -te)")
    parser.add_argument('-te', '--test_directory', default="", help="Directory with test data (if not set, use -tr)")
    parser.add_argument('-c', '--compile', action="store_true", help="If compile flearner (valid for type Wang/W)")
    parser.add_argument('--force', action="store_true", help="Force recomputing even if result file exists")
    parser.add_argument('--force_train', action="store_true",
                        help="Force traning even if result file exists (do not recompute features)")
    parser.add_argument('--dont_replace_fs', action="store_true", help="Do not replace filesystem when computing hash")
    parser.add_argument('-t', '--type', default="H",
                        help="Define method to use. Can be Wang/W, Panchenko/P or Hayes/H, Ensemble/E")
    parser.add_argument('-nt', '--no_type', default="",
                        help="Define method to not use. Can be Wang/W, Panchenko/P or Hayes/H")
    parser.add_argument('-pr', '--p_train', nargs='+', default=[], help="Train probability (incoming and outgoing)")
    parser.add_argument('-pri', '--p_train_in', nargs='+', default=['1'], help="Train probability (incoming)")
    parser.add_argument('-pro', '--p_train_out', nargs='+', default=['1'], help="Train probability (outgoing)")
    parser.add_argument('-pe', '--p_test', default=-1, type=float, help="Test probability (incoming and outgoing)")
    parser.add_argument('-pei', '--p_test_in', default=1, type=float, help="Test probability (incoming)")
    parser.add_argument('-peo', '--p_test_out', default=1, type=float, help="Test probability (outgoing)")
    parser.add_argument('-n', '--num_exps', default=1, type=int, help="Number of times the attack is made "
                                                                          "(returns averaged results)")
    parser.add_argument('-fi', '--first_index', type=int, default=0, help="First index of experiments.")
    parser.add_argument('-pd', '--p_dist', default="uniform", help="Probability distribution")
    parser.add_argument('-nci', '--num_consecutive_packets_in', default=20, type=int,
                        help="Number of incoming consecutive packets to send on same technology")
    parser.add_argument('-nco', '--num_consecutive_packets_out', default=20, type=int,
                        help="Number of outgoing consecutive packets to send on same technology")
    parser.add_argument('-nc', '--num_consecutive_packets', default=-1, type=int,
                        help="Number of outgoing and incoming consecutive packets to send on same technology")
    parser.add_argument('-nd', '--num_conspack_dist', default="exp",
                        help="Distribution of number of consective packets (fixed or exp)")
    parser.add_argument('-rri', '--round_robin_in', action="store_true",
                        help="If incoming packets sent in round-robin (only if splitting probability is 0.5)")
    parser.add_argument('-rro', '--round_robin_out', action="store_true",
                        help="If outgoing packets sent in round-robin (only if splitting probability is 0.5)")
    parser.add_argument('-rrr', '--round_robin_random', action="store_true",
                        help="If first technology in round robin is random")
    parser.add_argument('-cp', '--change_packets', default=0, type=int,
                        help="Number of packets after which splitting probability is changed")
    parser.add_argument('-ct', '--change_times', default=0, type=float,
                        help="Time after which splitting probability is changed")
    parser.add_argument('-cd', '--change_dist', default="fixed",
                        help="Distribution for changing time or packet (fixed or exp)")
    parser.add_argument('-ns', '--num_sites', default=100, type=int, help="Number of sites")
    parser.add_argument('-ni', '--num_insts', default=90, type=int, help="Number of instances")
    parser.add_argument('-no', '--num_open_insts', default=0, type=int, help="Number of open-world instances")
    parser.add_argument('-nf', '--test_prop', default=1/3, type=float, help="Proportion for test data")
    parser.add_argument('-nfo', '--test_prop_open', default=-1, type=float, help="Proportion for open-world test data")
    parser.add_argument('-nn', '--num_neighbors', type=int, help="Number of neighbors in kNN attack (Wang)")
    parser.add_argument('--deterministic_split', action="store_true",
                        help="If split train/test data deterministic (otherwise, randomly)")
    parser.add_argument('--single_thread', action="store_true", help="fextractor in single thread")
    parser.add_argument('-od', '--output_directory', default="out/out", help="Output folder")
    parser.add_argument('-of', '--output_file', default="attacks", help="Output log file")
    parser.add_argument('--only_in', action="store_true", help="Use only incoming traffic for attack")
    parser.add_argument('--only_out', action="store_true", help="Use only outgoing traffic for attack")
    parser.add_argument('-k', '--remove_ffiles', action="store_true", help="Remove f files before running")
    parser.add_argument('-fs', '--feature_stats', action="store_true", help="Show feature statistics")
    parser.add_argument('-m', '--multi_class', action="store_true",
                        help="If multi class accuracy with open world rather than two class")
    parser.add_argument('-v', '--verbose', default="INFO", help="Verbosity level (NONE, INFO, DEBUG, VDEBUG, ALL)")
    parser.add_argument('-ds', '--dont_save', action="store_true", help="Don't save flearner file")
    parser.add_argument('-ro', '--read_only', action="store_true", help="Only read result file (if it exists)")
    parser.add_argument('-dr', '--delay_sec_std_train', default=0, type=float,
                        help="Adds a delay (0-mean, arg is stddev) to every packet sent on secure channel (train)")
    parser.add_argument('-de', '--delay_sec_std_test', default=0, type=float,
                        help="Adds a delay (0-mean, arg is stddev) to every packet sent on secure channel (test)")
    parser.add_argument('-fp', '--factors_p_est', nargs='+', default=[], type=str,
                        help="Factors for p estimation")

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    if args.num_consecutive_packets > 0:
        args.num_consecutive_packets_in = args.num_consecutive_packets
        args.num_consecutive_packets_out = args.num_consecutive_packets
        if args.num_consecutive_packets == 1:
            args.num_conspack_dist = "fixed"

    args.light = light  # means that only file reading will be performed

    possible_attacks = ["Wang", "W", "Panchenko", "P", "Hayes", "H", "E"]

    if args.test_prop_open == -1:
        args.test_prop_open = args.test_prop

    if args.p_test >= 0:
        args.p_test_in = args.p_test
        args.p_test_out = args.p_test
    if len(args.p_train) > 0:
        args.p_train_str = ".".join(map(prob_to_zfill_str, args.p_train))
        args.p_train_in = args.p_train
        args.p_train_out = args.p_train
    else:
        if args.p_train_in == args.p_train_out:
            args.p_train_str = ".".join(map(prob_to_zfill_str, args.p_train_in))
        else:
            args.p_train_str = ".".join(map(prob_to_zfill_str,args.p_train_in+args.p_train_out))

    args.p_train_in = parse_p_trains(args.p_train_in)
    args.p_train_out = parse_p_trains(args.p_train_out)

    # logging
    logging.addLevelName(NONE, "NONE")
    logging.addLevelName(INFO, "INFO")
    logging.addLevelName(DEBUG, "DEBUG")
    logging.addLevelName(VDEBUG, "VDEBUG")
    logging.addLevelName(ALL, "ALL")
    logger = logging.getLogger('defenses')
    ch = logging.StreamHandler(sys.stdout)
    # Set logging format
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)
    # Set level format
    logger.setLevel(args.verbose)
    args.verbose = DICT_LOGS[args.verbose]
    args.logger = logger

    if args.type not in possible_attacks:
        print("Unknown attack type, can only be one of "+(" ".join(possible_attacks)))
        sys.exit(0)

    if args.p_test_in != args.p_test_out:
        p_test_str = prob_to_zfill_str(args.p_test_in)+"."+prob_to_zfill_str(args.p_test_out)
    else:
        p_test_str = prob_to_zfill_str(args.p_test_in)

    affix = ("-"+p_test_str+"-"+args.p_train_str)
    args.output_file += affix
    if args.output_directory == "/":
        args.output_directory = args.output_directory[:-1]
    args.output_directory += affix+"/"
    args.affix = affix

    if len(args.train_directories) == 0 and len(args.test_directory) == 0:
        print("Must provide at least test or train directories")
        sys.exit(0)

    for train_dir in args.train_directories:
        if not os.path.exists(train_dir) and not light:
            print("Train directory %s does not exist" % train_dir)
            sys.exit(0)

    if len(args.train_directories) == 0:
        args.train_directories = [args.test_directory]
    if len(args.test_directory) == 0:
        print("Using first of train directories as test directory")
        args.test_directory = args.train_directories[0]

    if not os.path.exists(args.test_directory) and not light:
        print("Test directory %s does not exist" % args.test_directory)
        sys.exit(0)

    if not light and not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    args.train_directories, args.test_directory, max_closed_sitenum, max_closed_instnum, max_open_sitenum, \
    args.first_closed_sitenum, _, _, args.delimiter, args.suffix\
        = check_data_format(args.train_directories, args.test_directory, light=args.light)

    if max_closed_sitenum > 0:
        args.num_sites = min(args.num_sites, max_closed_sitenum)
    if max_closed_instnum > 0:
        args.num_insts = min(args.num_insts, max_closed_instnum)
    if max_open_sitenum > 0:
        args.num_open_insts = min(args.num_open_insts, max_open_sitenum)

    if args.only_in and args.only_out:
        print("Cannot be both only_in and only_out")
        sys.exit(0)

    if args.change_times == 0 and args.change_packets == 0:
        args.change_dist = "fixed"

    if args.read_only:
        args.light = True

    if "all" in args.factors_p_est:
        args.factors_p_est = [1, 0.5, 0.33, 0.25, 2, 3, 4]
    else:
        args.factors_p_est = list(map(float, args.factors_p_est))

    return args


if __name__ == "__main__":

    args = get_args()

    if os.path.exists(args.test_directory[:-1]+".stats") and not args.force: # remove last slash
        f_stats = open(args.test_directory[:-1]+".stats", "r")
        for line in f_stats:
            print(line)

    if args.single_thread:
        n_jobs = 1

    attacks(args)
