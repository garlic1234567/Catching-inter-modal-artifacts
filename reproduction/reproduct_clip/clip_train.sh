EXPID=$(date +"%Y%m%d_%H%M%S")

HOST='127.0.0.1'
PORT='1'

NUM_GPU=2
python clip_train.py \
--config 'configs/train.yaml' \
--output_dir 'results' \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://${HOST}:1003${PORT} \
--token_momentum \
--world_size $NUM_GPU \
--model_save_epoch 10 \
