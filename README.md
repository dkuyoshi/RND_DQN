# RND_DQN

## Atari環境用です
[Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)

Random Network Distillation (RND)
+
Deep Q-Network


## Requirement

chainerrl==0.8.0

chainer==7.4.0

[chainerのgithub](https://github.com/chainer)



## Usage
`python train.py`

- --no_rnd : 普通のDQNでtrain(RNDを使用しない)
- --dueling :　Dueling Networkを用いる
- --gpu : デフォルトは-1(gpu使いたかったら0にする)

