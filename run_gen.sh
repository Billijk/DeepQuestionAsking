ckpt_name="eig"
data_path="./data/exp2"
model_opts="--energy_path ./energy --board_emb fix_binary --n_head 8 --n_layers 3 --d_k 32 --d_v 32"

directory="./checkpoints/$ckpt_name"
if [ ! -d "$directory" ]; then
    mkdir $directory
fi

if [ "$mode" == "train" ]; then
    train_opts="--lr 0.0001 --epoch 500 --save_epoch 100 --dropout 0.25 --reward eig"
    train_opts="$train_opts --batch_size 64 --step_penalty 0.05"
    log_opts="--log ./checkpoints/$ckpt_name/train.log"
    python train.py $data_path $directory $train_opts $model_opts $log_opts
elif [ "$mode" == "eval"]; then
    target=$2
    python test.py $data_path $target ${ckpt_name}.txt $model_opts
fi