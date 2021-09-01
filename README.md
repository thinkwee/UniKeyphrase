# UniKeyphrase
-   code for the ACL 2021 findings paper "UniKeyphrase: A Unified Extraction and Generation Framework for Keyphrase Prediction"
-   preprint paper: [arxiv](https://arxiv.org/pdf/2106.04847.pdf)
-   【2021.8.31】Update v2 version of the paper, no model code changed. See [v2](https://arxiv.org/abs/2106.04847) for the new version, [v1](https://arxiv.org/abs/2106.04847v1) for the older version
    -   fix a improper tokenization on the datasets which may lead to high results on present F1@M and low results on absent F1@5 & F1@M
    -   update some results of baseline(SEG-NET) from arxiv version to the newest ACL version
    -   following previous work, pad the result when calculating F1@5
    -   provide more detailed ablation study(layer and module)
    -   update table of "average numbers of predict keyphrases". UniKeyphrase now predicts more accurately after fixing the tokenization problem 
    -   update case study     

# Environment:
-   prepare for APEX
```
    . .bashrc
    apt-get update
    apt-get install -y vim wget ssh

    PWD_DIR=$(pwd)
    cd $(mktemp -d)
    git clone -q https://github.com/NVIDIA/apex.git
    cd apex
    git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
    python setup.py install --user --cuda_ext --cpp_ext
    cd $PWD_DIR
```
-   other packages
```
    pip install --user tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk
    python -c "import nltk; nltk.download('punkt')"
    pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval
```
-   get pretrained models from:  https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin

# Run
-   see run.sh and run_seq2seq.py

# Todo
-   data preprocess and postprocess script
-   quick start introduction
