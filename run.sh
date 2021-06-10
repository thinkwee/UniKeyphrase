MODEL_NAME=xxx
PROJECT_PATH=xxx
DATA_DIR=xxx
OUTPUT_DIR=xxx
MODEL_RECOVER_PATH=xxx/unilm1-base-cased.bin
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python run_seq2seq.py --do_train --fp16 --amp --num_workers 4 \
  --bert_model xxx/bert-base-cased \
  --new_segment_ids --tokenized_input \
  --data_dir ${DATA_DIR} --src_file kp20k.train.seq.in --tgt_file kp20k.train.absent \
  --label_file kp20k.train.seq.out \
  --output_dir ${OUTPUT_DIR} \
  --log_dir ${OUTPUT_DIR} \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 384 --max_position_embeddings 384 \
  --max_len_b 32 \
  --mask_prob 0.7 --max_pred 32 \
  --train_batch_size 256 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 100 --use_SRL --use_bwloss