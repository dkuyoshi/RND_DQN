# RND_DQN

## Atari環境用です
[Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)

Random Network Distillation (RND)
+
Deep Q-Network
(元論文はPPOですが...)
## TODO

- [x] model
- [x] prediction
- [x] target
- [x] 報酬追加 <- target計算
- [x] predictionのtrain

## チューニングが厄介？
- [ ] reward normalization : In order to keep the rewards on a consistent scale we normalized the intrinsic reward by dividing it by a running estimate of the standard deviations of the intrinsic returns. 
- [ ]  observation normalization : we whiten each dimension by subtracting the running mean and then dividing by the running standard deviation. We then clip the normalized observations to be between -5 and 5. We initialize the normalization parameters by stepping a random agent in the environment for a small number of steps before beginning optimization. We use the same observation normalization for both predictor and target networks but not the policy network.

![Hyper](https://github.com/dkuyoshi/RND_DQN/blob/master/images/image.png "Hyperparameter for normalization")
[Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)より引用

などいっぱい