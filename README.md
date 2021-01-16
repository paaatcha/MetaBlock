# MetaBlock repository

In this repository I share the implemented code for the MetaBlock approach. I provide some intruction of how to use it in the following.
If you find any bug or have any observation, please, let me know. Don't hesitate to get in touch with me.


## Dependencies
All code is done in Python enviroment. Machine/Deep leaning models are implemented using [Scikit-Learn](https://scikit-learn.org/stable/) and [Pytorch](https://pytorch.org/).

If you already have the Python enviroment set in your machine, and you're using Linux (just like me), all you need to do to install all dependencies is to run `pip install -r requirements.txt`

If you're in the wrong side of the force and are using Windows, I suggest you to use [Anaconda](https://www.anaconda.com/) to set up your enviroment. However, I don't know if everything will work properly.

## Raug repository
To run this code you must clone the [Raug](https://github.com/paaatcha/raug) repository from my Github. Raug is reposible to train the deep learning models. You may find mode instruction on its own `Readme.md`

After cloning this repository, you must set the path on `constants.py` file. You're going to find instructions there.

## Organization
- In `my_models` folder you're going to find the CNN models implementations as well as the concatenation approach, the MetaNet, and the MetaBlock.
- In `benchmarks` folder are the scripts for the experiments in ISIC 2019 and PAD-UFES-20 datasets.

To run the benchmarks, I used [Sacred](https://sacred.readthedocs.io/en/stable/index.html), which is basically a tool to organize experiments.
You don't need to kown how to use it in order to run the code, although I strong recommend you to learn it.

Using Sacred, you may run an experiment in the following way:
`python pad.py with _lr=0.001 _batch_size=50`

If you don't want to use it, you can change the parameters directly in the code.

**Important**: you must set the path of the dataset in `benchmark/pad.py` and `benchmark/isic.py`


## Where can I find the datasets?
You may find the link to all datasets I used in my thesis in the following list:
- [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2)
- [ISIC 2019](https://challenge2019.isic-archive.com/)
