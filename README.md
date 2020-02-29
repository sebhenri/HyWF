# HyWF

This folder contains the code and dataset used for the paper Protecting against Website Fingerprinting with Multihoming, published at PoPETS 2020.

This code requires python 3.5.

To install the required packages, run
`pip install -r requirements.txt`
This can be done for example in a virtual environment through the following
```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

This repository contains two folders:

  * `HyWF_code` contains the code used for our paper. It requires traces with a timestamp and a direction per packet, as is the case for most of the publicly available datasets (see Section 4.1 of our paper).

  * `HyWF_dataset` contains the multipath dataset described in Section 7 of our paper. Datasets for the different scenarios are provided as zipped tarballs and should be unzipped before running the code.

Corresponding READMEs can be found in `HyWF_code` and `HyWF_dataset`.

