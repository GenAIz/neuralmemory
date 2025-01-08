# MANN-RL
Memory-augmented neural networks (MANN) are a subset of neural networks that have a read-write memory which can allow the network to access past information other than its current inputs. A new MANN architecture is explored herein.  This MANN architecture, called MANN-RL, uses reinforcement learning (RL) to learn what should be remembered.  The MANN-RL architecture defines a neural network called the memory-net which produces the memory at time 't + 1' given the memory at time 't'.  This repository contains the code for the MANN-RL architecture and its application to a sequential logic problem, part-of-speech tagging and sentence memorization.  MANN-RL performed well (0.97 F1 and above) on three relatively simple sequence-based problems.


# Python environment set up
The commands to setup the python environment are below.

The python version is `3.11.10`.

```commandline
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip==23.3.2
pip install torch torchtext torchvision torchaudio torchdata
pip install 'portalocker>=2.0.0'
pip install -U scikit-learn
pip install PyYAML
pip install bpemb
```

Pip freeze gives the following:
```
bpemb==0.3.6
certifi==2023.11.17
charset-normalizer==3.3.2
filelock==3.13.1
fsspec==2023.12.2
gensim==4.3.3
idna==3.6
Jinja2==3.1.2
joblib==1.3.2
MarkupSafe==2.1.3
mpmath==1.3.0
networkx==3.2.1
numpy==1.26.3
pillow==10.2.0
portalocker==2.8.2
PyYAML==6.0.1
requests==2.31.0
scikit-learn==1.3.2
scipy==1.11.4
sentencepiece==0.2.0
smart-open==7.0.5
sympy==1.12
threadpoolctl==3.2.0
torch==2.1.2
torchaudio==2.1.2
torchdata==0.7.1
torchtext==0.16.2
torchvision==0.16.2
tqdm==4.66.1
typing_extensions==4.9.0
urllib3==2.1.0
wrapt==1.16.0
```

From the root of this repository `export PYTHONPATH=$PWD/src`.

From the root of this repository `export NM_DATA_DIR=$PWD/data`

# Unit tests
There is a __minimal__ set of unit tests.  These can validate that that python environment is likely set up correctly.

To run the unit tests: `python -m unittest discover -s tests`.


# Experiments

To start an experiment run a command following the template `python src/nm/train.py <config yaml file> <experiment output directory>`.  Example experiment configuration files are found in the `config` directory.  The logic configs are in `config/logic`.  The POS configs are in `config/pos`.  The sentence memorization and machine translation configs are in `config/mt`.

An example: `python src/nm/train.py config/logic/logic_m_best.yaml experiments/test`

## Experimental results

If everything is setup properly and runs without issue then one should expect to achieve 0.99, 0.97 and 0.97 F1 for the logic, POS and sentence memorization problems respectively.

## Data sets
Please see each individual data set for its license, etc.  The sentence data is in `wmt2014.zip`.  Original wmt 2014 news data set can be obtained from https://www.statmt.org/wmt14/translation-task.html and placed in `data/wmt2014-news`, giving:
```
$ ls data/wmt2014-news
news-commentary-v9.cs-en.cs	news-commentary-v9.de-en.de	news-commentary-v9.fr-en.en	news-commentary-v9.ru-en.en	training-parallel-nc-v9.tgz
news-commentary-v9.cs-en.en	news-commentary-v9.de-en.en	news-commentary-v9.fr-en.fr	news-commentary-v9.ru-en.ru
```


# Citation

If you would like to cite this work then you may use:
*** TODO
