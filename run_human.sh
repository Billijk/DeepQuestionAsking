ckpt_name="eig"
data_path="./data/exp1"
model_opts="--energy_path ./energy --board_emb fix_binary --n_head 8 --n_layers 3 --d_k 32 --d_v 32"

directory="./checkpoints/$ckpt_name"
if [ ! -d "$directory" ]; then
    mkdir $directory
fi

mode=$1
if [ "$mode" == "pretrain" ]; then
    pretrain_opts="--lr 0.001 --epoch 200 --save_epoch 100 --dropout 0.5 --reward eig --batch_size 1024"
    log_opts="--log ./checkpoints/$ckpt_name/pretrain.log"
    python pretrain.py $data_path $directory $pretrain_opts $model_opts $log_opts
elif [ "$mode" == "validation"]; then
    train_opts="--lr 0.001 --epoch 15 --dropout 0.25"
    python cv.py $data_path ./checkpoints/pretrain/ep_200.pth cv_output.txt $train_opts $model_opts
fi