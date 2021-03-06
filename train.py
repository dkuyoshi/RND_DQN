import argparse
import os

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DuelingDQN
from chainerrl import replay_buffer

from chainerrl.wrappers import atari_wrappers

from q_function import DQNQFunction, DuelingQFunction
from agent import RNDAgent
from rnd_network import RNDModel
from train_agent import train_agent_with_evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='OpenAI Atari domain to perform algorithm on.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6,
                        help='Timesteps after which we stop ' +
                        'annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.01,
                        help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.001,
                        help='Exploration epsilon used during eval episodes.')
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--arch', type=str, default='doubledqn',
                        choices=['nature', 'nips', 'dueling', 'doubledqn'],
                        help='Network architecture to use.')
    parser.add_argument('--steps', type=int, default=10 ** 7,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5*10**4,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--target-update-interval',
                        type=int, default=3 * 10 ** 4,
                        help='Frequency (in timesteps) at which ' +
                        'the target network is updated.')
    parser.add_argument('--eval-interval', type=int, default=10 ** 5,
                        help='Frequency (in timesteps) of evaluation phase.')
    parser.add_argument('--update-interval', type=int, default=4,
                        help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.add_argument('--num-step-return', type=int, default=1)
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate.')
    parser.add_argument('--prioritized', action='store_true', default=False,
                        help='Use prioritized experience replay.')
    parser.add_argument('--checkpoint-frequency', type=int,
                        default=None,
                        help='Frequency at which agents are stored.')
    parser.add_argument('--dueling', action='store_true', default=False,
                        help='use dueling dqn')
    parser.add_argument('--normalization_pre_steps', type=int, default=5*10**3,
                        help='steps for initializing the normalization parameters')
    parser.add_argument('--no_rnd', action='store_true', default=False,
                        help='simple dqn training')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    if args.dueling:
        name = 'DuelingDQN'
        if args.no_rnd:
            name = 'SimpleDuelingDQN'
    else:
        name = 'DQN'
        if args.no_rnd:
            name = 'SimpleDQN'

    args.outdir = experiments.prepare_output_dir(args, args.outdir, time_format='{}/{}/%Y%m%dT%H%M%S.%f'.format(args.env, name))
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    #q_func = parse_arch(args.arch, n_actions)
    if args.dueling:
        q_func = DuelingQFunction(n_actions,)
    else:
        q_func = DQNQFunction(n_actions,)

    rnd = RNDModel(gpu=args.gpu)
    #rnd_func = rnd.predict
    if args.noisy_net_sigma is not None:
        links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()
    else:
        explorer = explorers.LinearDecayEpsilonGreedy(
            1.0, args.final_epsilon,
            args.final_exploration_frames,
            lambda: np.random.randint(n_actions))

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # Adam
    opt = optimizers.Adam()
    opt_rnd = optimizers.Adam()

    opt.setup(q_func)
    opt_rnd.setup(rnd.predict)

    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            10 ** 6, alpha=0.6,
            beta0=0.4, betasteps=betasteps,
            num_steps=args.num_step_return)
    else:
        rbuf = replay_buffer.ReplayBuffer(10 ** 6, args.num_step_return)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    def phi_initial(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32)

    if not args.no_rnd:
        Agent = agents.DQN
        agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                      explorer=explorer, replay_start_size=args.replay_start_size,
                      target_update_interval=args.target_update_interval,
                      clip_delta=args.clip_delta,
                      update_interval=args.update_interval,
                      batch_accumulator='sum',
                      phi=phi)
    else:
        Agent = RNDAgent
        agent = Agent(q_func, rnd, opt, opt_rnd, rbuf, gpu=args.gpu, gamma=0.99,
                      gamma_i=0.99,
                      explorer=explorer, replay_start_size=args.replay_start_size,
                      target_update_interval=args.target_update_interval,
                      clip_delta=args.clip_delta,
                      update_interval=args.update_interval,
                      batch_accumulator='mean',
                      phi=phi,
                      phi_i=phi_initial,
                      pre_steps=args.normalization_pre_steps,
                      n_action=n_actions,)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        if not args.no_rnd:
            experiments.train_agent_with_evaluation(
                agent=agent, env=env, steps=args.steps,
                eval_n_steps=None,
                checkpoint_freq=args.checkpoint_frequency,
                eval_n_episodes=args.eval_n_runs,
                eval_interval=args.eval_interval,
                outdir=args.outdir,
                save_best_so_far_agent=False,
                eval_env=eval_env,
            )
        else:
            train_agent_with_evaluation(
                agent=agent, env=env, steps=args.steps,
                eval_n_steps=None,
                checkpoint_freq=args.checkpoint_frequency,
                eval_n_episodes=args.eval_n_runs,
                eval_interval=args.eval_interval,
                outdir=args.outdir,
                save_best_so_far_agent=False,
                eval_env=eval_env,
            )

if __name__ == '__main__':
    main()