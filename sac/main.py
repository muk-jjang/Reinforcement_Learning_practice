import os
# Fix for headless server
# os.environ['MUJOCO_GL'] = 'osmesa'  # Commented out for macOS compatibility
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/usr/local/mujoco'

import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from replay_buffer import ReplayBuffer
import wandb

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v4",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True

wandb.init(project="sac-v4", name=f"{args.env_name}-{args.policy}-{args.seed}")

#Environment
env = gym.make(args.env_name)
# env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Debug prints
# print(f"Environment: {args.env_name}")
# print(f"Observation space: {env.observation_space}")
# print(f"Observation space shape: {env.observation_space.shape}")
# print(f"Action space: {env.action_space}")
# print(f"Action space shape: {env.action_space.shape}")
# print(f"Hidden size: {args.hidden_size}")

#Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
# buffer
memory = ReplayBuffer(args.replay_size, args.seed)

# Training
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    
    if isinstance(state, tuple):
        state = state[0]
    
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
        
        if len(memory) > args.batch_size:
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
            # logging이 너무 자주 이루어짐 -> 1000 timestep마다 logging
            if total_numsteps % 1000 ==0:
                wandb.log({
                    "loss/qf1_loss": critic_1_loss,
                    "loss/qf2_loss": critic_2_loss,
                    "loss/policy_loss": policy_loss,
                    "loss/ent_loss": ent_loss,
                    "entropy_temprature": alpha,
                })
            updates += 1
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        total_numsteps += 1
        episode_steps += 1
    
        # Handle next_state if it's a tuple (for newer gym versions)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        memory.push(state, action, reward, next_state, done)
        state = next_state

    if total_numsteps > args.num_steps:
        break

    wandb.log({
        'reward/train': episode_reward,
        'time/episode_num': i_episode
        })
    
    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state = env.reset()
            
            # Handle state if it's a tuple (for newer gym versions)
            if isinstance(state, tuple):
                state = state[0]
                
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Handle next_state if it's a tuple (for newer gym versions)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                    
                state = next_state

            avg_reward += episode_reward

        avg_reward /= episodes
        wandb.log({
            'reward/test': avg_reward,
            'time/episode_num': i_episode
        })
env.close()