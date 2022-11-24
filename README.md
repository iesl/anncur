# Efficient Nearest Neighbor Search for Cross-Encoder Models using Matrix Factorization
This repository contains code used in experiments for our EMNLP 2022 paper titled  "[Efficient Nearest Neighbor Search for Cross-Encoder Models using Matrix Factorization](https://arxiv.org/pdf/2210.12579.pdf)". 

## Setup ##

* Clone the repository and install the dependencies (optionally) in a separate conda environment.
```
conda create -n <env_name> -y python=3.7 && conda activate <env_name>
pip install -r requirements.txt
```

* Setup some enviroment variables

```
source bin/setup.sh
```

## Data Setup ##
1. Download [ZeShEL](https://paperswithcode.com/dataset/zeshel) data from [here](https://github.com/lajanugen/zeshel).
2. Preprocess data into the required format using `utils/preprocess_zeshel.py` in order to train dual-encoder and 
cross-encoder models on this dataset. We will use standard train/test/dev splits as defined [here](https://aclanthology.org/P19-1335.pdf).


## Training Dual-Encoder and Cross-Encoder Models ##

To train a dual-encoder model, run

```
python models/train.py --config config/el_zeshel_bi_enc.json
```

To train a cross-encoder model (using dual-encoder checkpoint for mining negatives), run

```
python models/train.py --config config/el_zeshel_cross_enc.json --neg_mine_bienc_model_file <path to a biencoder model checkpoint>
```

Note that these config files expect data to be present in `../../data/zeshel` folder and 
trained models are saved in `../../results` folder. 

Finally, trained models can be evaluated for the task of entity linking using `eval/run_cross_encoder_w_binenc_retriever_zeshel.py`. 

## Nearest Nbr Search Exps ##
For running nearest neighbor search experiments, we first need to compute exact scores using
a cross-encoder model. For this dataset, queries correspond to mentions of entities in context and 
items correspond to entity titles with their descriptions.

Exact query-item score matrices can be computed using `eval/run_cross_encoder_for_ment_ent_matrix_zeshel.py` file. 

All methods are evaluated on the task of finding top-k nearest neighbors items for a given query 
using `eval/run_retrieval_eval_wrt_exact_crossenc.py` and `eval/run_retrieval_eval_wrt_exact_crossenc_w_fixed_train_test_splits.py`.


## Download pre-trained dual-encoder and cross-encoder models used in the paper from huggingface

* [Dual-Encoder Model](https://huggingface.co/nishantyadav/dual_encoder_zeshel)
* [Cross-Encoder Model w/ [CLS] token pooling](https://huggingface.co/nishantyadav/cls_crossencoder_zeshel)
* [Cross-Encoder Model w/ proposed special token based pooling (see paper for details)](https://huggingface.co/nishantyadav/emb_crossenc_zeshel)
