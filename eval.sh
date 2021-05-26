model_opts="--energy_path ./energy --board_emb fix_binary --n_head 8 --n_layers 3 --d_k 32 --d_v 32"

mode=$1
target=$2
elif [ "$mode" == "perplexity"]; then
    data_path="./data/exp1"
    python perplexity.py $data_path $target $eval_opts $model_opts
elif [ "$mode" == "reinforce"]; then
    data_path="./data/exp2"
    python test.py $data_path $target ${ckpt_name}.txt $eval_opts $model_opts
fi