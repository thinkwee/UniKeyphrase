PROJECT_PATH=xxxx
DATA_DIR=xxxx
MODEL_RECOVER_PATH=xxxx
RESULT_DIR=xxxx
EVAL_SPLIT=kp20k.test
export CUDA_VISIBLE_DEVICES=0
python biunilm/decode_seq2seq.py --fp16 --amp \
  --bert_model xxxx/bert-base-cased \
  --new_segment_ids --mode s2s --need_score_traces \
  --input_file ${DATA_DIR}/${EVAL_SPLIT}.seq.in --split ${EVAL_SPLIT} --tokenized_input \
  --output_file xxxx \
  --output_label_file xxxx \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_tgt_length 32 \
  --batch_size 16 --beam_size 5 --length_penalty 0 \
  --forbid_duplicate_ngrams --forbid_ignore_word "."
