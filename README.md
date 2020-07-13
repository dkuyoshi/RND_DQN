# RND_DQN

## Atari環境用です
[Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)

Random Network Distillation (RND)
+
Deep Q-Network
(元論文はPPOですが...)

## Requirement

chainerrl==0.8.0

chainer==7.4.0

[chainerのgithub](https://github.com/chainer)
## TODO

- [x] model
- [x] prediction
- [x] target
- [x] 報酬追加 <- target計算
- [x] predictionのtrain

## チューニングが厄介？
- [ ] reward normalization : In order to keep the rewards on a consistent scale we normalized the intrinsic reward by dividing it by a running estimate of the standard deviations of the intrinsic returns.(報酬を一定のスケールに保つために、固有のリターンの標準偏差の現在の推定値で割ることにより、固有の報酬を正規化報酬を一定のスケールに保つために、固有のリターンの標準偏差の現在の推定値で割ることにより、固有の報酬を正規化)
- [ ]  observation normalization : we whiten each dimension by subtracting the running mean and then dividing by the running standard deviation. We then clip the normalized observations to be between -5 and 5. We initialize the normalization parameters by stepping a random agent in the environment for a small number of steps before beginning optimization. We use the same observation normalization for both predictor and target networks but not the policy network.(実行中の平均を差し引き、次に実行中の標準偏差で割ることにより、各次元を白くします。 次に、正規化された観測値を-5から5の間になるようにクリップします。最適化を開始する前に、環境内でランダムエージェントを少ないステップ数だけステップ実行して、正規化パラメーターを初期化します。 予測ネットワークとターゲットネットワークの両方に同じ観測正規化を使用しますが、ポリシーネットワークは使用せず。)

![Hyper](https://github.com/dkuyoshi/RND_DQN/blob/master/images/image.png "Hyperparameter for normalization")
[Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)より引用

などいっぱい