MODEL_NAME=xxxx
PROJECT_PATH=xxxx
DATA_DIR=xxxx
OUTPUT_DIR=xxxx
MODEL_RECOVER_PATH=xxxx
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biunilm/run_seq2seq.py --do_train --amp --num_workers 4 \
  --bert_model xxxx/bert-base-cased \
  --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} --src_file kp20k.train.seq.in --tgt_file kp20k.train.absent \
  --label_file kp20k.train.seq.out \
  --output_dir ${OUTPUT_DIR} \
  --log_dir ${OUTPUT_DIR} \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_position_embeddings 192 \
  --max_len_b 32 \
  --mask_prob 0.7 --max_pred 32 \
  --train_batch_size 256 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 100
