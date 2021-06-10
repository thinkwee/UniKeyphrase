# UniKeyphrase
-   code for the ACL 2021 findings paper "UniKeyphrase: A Unified Extraction and Generation Framework for Keyphrase Prediction"
-   preprint paper: [arxiv](https://arxiv.org/pdf/2106.04847.pdf)

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
