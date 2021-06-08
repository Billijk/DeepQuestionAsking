# Modeling Question Asking Using Neural Program Generation

Code for our CogSci 2021 paper *Modeling Question Asking Using Neural Program Generation* ([ArXiv](https://arxiv.org/abs/1907.09899)).

## Requirements:
* Python3
* PyTorch >= 1.1.0
* [expected-information-gain](https://github.com/anselmrothe/EIG)

## How to run our code
### Estimating the distribution of human questions
This needs to pre-train the model on the synthesized dataset and the fine-tune on the human questions.
```bash
./run_human.sh pretrain
```

For cross-validation, run
```bash
./run_human.sh validation
```

### Question generation
```bash
./run_gen.sh train
```

For inference and evaluation, run
```bash
./run_gen.sh eval ./checkpoints/ep_500.pth
```

## Acknowledgement
This work was supported by Huawei. We are grateful to Todd Gureckis and Anselm Rothe for helpful comments and conversations. We thank Jimin Tan for writing the initial version of the RL-based training codes.

## Citation
If you use our codes, please kindly cite our paper.
```bibtex
@inproceedings{wang2021modeling,
  title={Modeling Question Asking Using Neural Program Generation},
  author={Wang, Ziyun and Lake, Brenden},
  booktitle={Proceedings of CogSci 2021},
  year={2021}
}
```
