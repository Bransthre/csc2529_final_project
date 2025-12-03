# Incremental DIPDefend

This is a final project for CSC2529, and the work is heavily based on the following works.
```
@inproceedings{DIPDefend_2020,
  title       = {DIPDefend: Deep Image Prior Driven Defense against Adversarial Examples},
  author      = {Tao Dai, Yan Feng, Dongxian Wu, Bin Chen, Jian Lu, Yong Jiang, Shu-Tao Xia},
  booktitle   = {Proceedings of the 28th ACM International Conference on Multimedia},
  year        = {2020}
}
@article{UlyanovVL17,
  author      = {Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor},
  title       = {Deep Image Prior},
  journal     = {arXiv:1711.10925},
  year        = {2017}
}
```
As well as [`marcusGH`'s work here](https://github.com/marcusGH/adversarial-attacks-on-imagenet-models/tree/main).
This is an independent course project.

## Installations
First of all, the work uses the mini-ImageNet dataset as noted [here](https://www.kaggle.com/datasets/zcyzhchyu/mini-imagenet). You will need to position your dataset in the following format:
```
\root
  \dataset
    \mini-imagenet
      \images  // (decompress images.tar here)
      \labels
        |- test.csv
        |- train.csv
        |- val.csv
```

This work uses Python 3.11 and the required packages can be downloaded via:
```
pip install -r requirements.txt
```

## Launching Experiments
Once you have installed your virtual environments as instructed above, you can
launch examplar experiments via the following line:
```
bash script_launchpoint/launch_all_defenses.sh
```
Note that the script is defaulted to run on a SLURM-based system, and is
specifically tailored to some customs in [Compute Canada (CCDB)](https://docs.alliancecan.ca/wiki/Getting_started) devices.

All analyses are conducted via code under `script_analyses`, but some will only successfully run after you acquire experiment data by running the above script.

## Notes for Grading
To protect the grade of implementation, I think it's necessary to make some notes about the "out-of-the-box"-ness of this codebase.

First of all, as for most machine learning repositories, running implementations of any publication always requires some adaptation of their codebase into your own machine. Generally, the codebase runs out-of-the-box with some deserved adjustments to your own GPU cluster. For example, changing the virtual environment names prescribed in some bash scripts.

Additionally, existing experiment results are not attached in this repository per GitHub file size regulation. Without having launched experiments from the aforementioned scripts, you should not expect analyses scripts to run.

If you are interested in launching the experiments, this work is trained on L40 GPUs provided by the Killarney cluster from computecanada, accessible for Vector Institute affiliates at the time of writing. Each experiment's expected time is as stated in the bash scripts.
