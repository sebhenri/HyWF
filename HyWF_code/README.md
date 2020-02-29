# HyWF_code

This folder contains the code used for the paper Protecting against Website Fingerprinting with Multihoming, published at PoPETS 2020.

The different options can be found in the files or with the -h option (e.g., `python attacks.py -h`)

The two main files are attacks.py and defenses.py.

1. attacks.py enables us to run several attacks: k-fingerprinting, k-NN, and CUMUL.
   We reused the code provided by the authors of the following attacks:

  1. k-fingerprinting (Hayes et al.), attack used with argument `-t H`
  2. k-NN (Wang et al.), attack used with argument `-t W`

   This code also implements CUMUL (Panchenko et al.), used with argument `-t P`

Our paper also uses the attack Deep-Fingerprinting (Sirinam et al.) for which we reused directly the authors' code and which is not contained in this code.

2. defenses.py enables us to run several defenses. We reused the code provided by the authors of the following defense:
  * WTFPAD (Juarez et al.), used with argument `-t W`.

Examples:
`python attacks.py -tr ../HyWF_dataset/mptcp-hywf_wifiwifi_cells/p1_cells -t H`
enables us to run the k-fingerprinting attack on path 1 of the MPTCP-HyWF dataset (exp dataset in the paper, obtained
with MPTCP and HyWF scheduler; the TPR should be around 37%).

`python attacks.py -tr ../HyWF_dataset/tcp-wifi_cells -t H`
enables us to run the k-fingerprinting attack on single-path TCP (exp dataset in the paper, obtained with single-path TCP;
the result should be around 90%).

Our defense HyWF can be evaluated directly with original trace by running attacks.py with the -pr (average splitting
probability of the training data) and -pe (average splitting probability of the test data) arguments set to a value
smaller than 1.
To evaluate HyWF on the knn dataset (not provided here) with traces repeated 5 times (n_pr = 30,000 in the paper), one can run
`python attacks.py -tr ../HyWF_dataset/knn -t H -pr 5*0.5 -pe 0.5`
By default, when the splitting probability -pr is smaller than 1, the parameters used are the parameters that define HyWF in the paper.

`dump_Alexa_HS` and `dump_knn` contain the histograms used for WTF-PAD defense.

The attack requires some information about the datasets (e.g., number of sites, etc.). These information are automatically computed (see `check_data_format` in `utils.py`) and saved in a format.pickle file to make the next attack faster.

Results of an attack are saved in the out/ folder. The next run of attack.py with the next parameters simply loads the results. `--force` can be used to force the attack to be run again from scratch.

