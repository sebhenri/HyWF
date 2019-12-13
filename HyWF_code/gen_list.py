#generate trainlist and testlist for other files
import sys
import glob
import re
import random
import numpy as np
from pprint import pprint
from constants import *
from utils import *


def gen_list(args):

    train_directories, test_directory, max_closed_sitenum, max_closed_instnum, max_open_sitenum, \
    first_closed_sitenum, first_closed_instnum, first_open_sitenum, delimiter, suffix = \
        check_data_format(args.train_directories, args.test_directory, light=args.light)

    if args.light:
        return train_directories, test_directory, [], [], []

    # --------- split train/test -----------

    # write in file for Wang attack
    trainout = open(args.output_directory + "trainlist", "w")
    testout = open(args.output_directory + "testlist", "w")

    test_files = []
    train_files = []
    weight_files = []
    all_files = []
    # closed-world (monitored)
    for s in range(first_closed_sitenum, args.num_sites+first_closed_sitenum):
        # choose test files
        r_ints = range(first_closed_instnum, args.num_insts+first_closed_instnum)
        if args.deterministic_split:
            test_inds = list(range(0, int(np.ceil(len(r_ints)*args.test_prop))))
        else:
            test_inds = random.sample(r_ints, int(np.ceil(len(r_ints)*args.test_prop)))

        for i in r_ints:
            sname = args.cellf_loc + str(s) + delimiter + str(i) + suffix
            if i in test_inds:
                test_files.append(sname.split("/")[-1])
                testout.write(sname + "\n")
            else:
                train_files.append(sname.split("/")[-1])
                weight_files.append(sname.split("/")[-1])
                trainout.write(sname + "\n")
            all_files.append(sname)

    # open-world (unmonitored)
    r_ints = range(first_open_sitenum, args.num_open_insts)
    if args.deterministic_split:
        test_inds = range(0, int(len(r_ints)*args.test_prop_open))
    else:
        test_inds = random.sample(r_ints, int(len(r_ints)*args.test_prop_open))
    for s in range(first_open_sitenum, args.num_open_insts):
        sname = args.cellf_loc + str(s) + suffix
        if s in test_inds:
            test_files.append(sname.split("/")[-1])
            testout.write(sname + "\n")
        else:
            train_files.append(sname.split("/")[-1])
            trainout.write(sname + "\n")
        all_files.append(sname)

    trainout.close()
    testout.close()

    if args.verbose <= VDEBUG:
        print(test_files)

    wout = open(args.output_directory + "weightlist", "w")
    #all_files = sorted(glob.glob(directory + "*")) # if include open world for closed world
    weight_files = [sname.split("/")[-1] for sname in weight_files]
    [wout.write(args.cellf_loc+sname.split("/")[-1] + "\n") for sname in weight_files]
    wout.close()

    print("Generated lists with %d test files, %d train files" % (len(test_files), len(train_files)))

    return train_directories, test_directory, sorted(test_files), sorted(train_files), sorted(weight_files)
