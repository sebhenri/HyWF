# HyWF_code

This folder contains the code used for the paper Protecting against Website Fingerprinting with Multihoming, published at PoPETS 2020.

The different options can be found in the files or with the -h option (e.g., python attacks.py -h)

The two main files are attacks.py and defenses.py.

attacks.py enables us to run several attacks: k-fingerprinting, k-NN, and CUMUL.
We reused the code provided by the authors of the following attacks:
- k-fingerprinting (Hayes et al.), attack used with argument -t H
- k-NN (Wang et al.), attack used with argument -t W
This code also implements CUMUL (Panchenko et al.), used with argument -t P
Our paper also uses the attack Deep-Fingerprinting (Sirinam et al.) for which we reused directly the authors' code.

defenses.py enables us to run several defenses. We reused the code provided by the authors of the following defense:
- WTFPAD (Juarez et al.), used with argument -t W.

Examples:
python attacks.py -tr ../HyWF_datasets/mptcp-hywf_wifiwifi_cells/p1_cells -t H
enables us to run the k-fingerprinting attack on path 1 of the MPTCP-HyWF dataset (exp dataset in the paper, obtained
with MPTCP and HyWF scheduler; the TPR should be around 37%).
python attacks.py -tr ../HyWF_datasets/tcp-wifi_cells -t H
enables us to run the k-fingerprinting attack on single-path TCP (exp dataset in the paper, obtained with single-path TCP;
the result should be around 90%).

Our defense HyWF can be evaluated directly with original trace by running attacks.py with the -pr (average splitting
probability of the training data) and -pe (average splitting probability of the test data) arguments set to a value
smaller than 1.
To evaluate HyWF on the knn dataset (not provided here) with traces repeated 5 times (n_pr = 30,000 in the paper), one can run
python attacks.py -tr ../HyWF_datasets/knn -t H -pr 5*0.5 -pe 0.5
By default, when the splitting probability -pr is smaller than 1, the parameters used are the parameters that define HyWF in the paper.

dump_Alexa_HS and dump_knn contain the histograms used for WTF-PAD defense.