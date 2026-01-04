#
#   Entry point script, for training and testing.
#

import gymnasium as gym
from dqn import TrainingEvaluator
from dqn import TrainerDQN, TrainerDQNHyperparams
from gymnasium.wrappers import FrameStackObservation

from flappy_nn import FlappyBirdCNN
from flappy_env_wrapper import FlappyBirdImgObservation

import flappy_bird_gymnasium

import pygame
import numpy as np

import argparse

from datetime import datetime
import os
import sys

import pandas as pd

def train_flappy(dirpath, hyperparams_path, steps):
    env = gym.make("FlappyBird-v0", max_episode_steps=1000, render_mode="rgb_array", use_lidar = False, background=None)
    env = FlappyBirdImgObservation(env)

    eval_env = gym.make("FlappyBird-v0", max_episode_steps=1000, render_mode="rgb_array", use_lidar = False, background=None)
    eval_env = FlappyBirdImgObservation(env)

    eval_env = FrameStackObservation(eval_env, stack_size=4, padding_type="zero")

    callback = TrainingEvaluator(eval_env=eval_env, eval_freq=8, runs_per_eval=5, rollout_freq = 4, progress_bar=False)

    new_hyperparams = False
    if hyperparams_path is not None:
        hyperparameters = TrainerDQNHyperparams.from_file(hyperparams_path)
        new_hyperparams = True
    else:
        hyperparameters = TrainerDQNHyperparams(
                learning_rate=0.0002,
                exploration_fraction=0.2,
                exploration_rate_initial=1.0,
                exploration_rate_final=0.03,
                replay_buffer_size = 100_000,
                learning_starts = 10_000,
                gamma = 0.99,
                batch_size = 32,
                target_update_interval=512,
                train_freq = 4,
                frame_stack = 4,
                double_dqn = True,
                dueling_dqn =True
        )

    if dirpath is not None:     
        dqn = TrainerDQN.load_from_file(obs_space_shape=env.observation_space.shape,
                                    obs_space_dtype=env.observation_space.dtype,
                                    action_space_dim=env.action_space.n,
                                    filepath=os.path.join(dirpath, "dqn.zip"),
                                    model_class=FlappyBirdCNN,
                                    new_hyperparams=(hyperparameters if new_hyperparams else None), # use new hyperparams if defined
                                    use_cuda_device=True,
                                    callback=callback,
                                    mode="train")
    else:
        dqn = TrainerDQN(
            obs_space_shape=env.observation_space.shape,
            obs_space_dtype=env.observation_space.dtype,
            action_space_dim=env.action_space.n,
            hyperparameters = hyperparameters,
            model_class=FlappyBirdCNN,
            model_kwargs=dict(features_dim=256, filters=[16, 32]),
            use_cuda_device=True,
            callback = callback
        )

    try:
        dqn.learn(env= env, target_timesteps=steps)
    except KeyboardInterrupt:
        print("Keyboard interrup. Stopping early.")
    finally:
        if callback.is_worth_saving():
            print("Saving data...")

            formatted_time = datetime.now().strftime("%d-%m_%H-%M")
            train_dirname = f"flappy_{formatted_time}_{os.getpid()}"

            os.makedirs(train_dirname, exist_ok=True)

            dqn.save(os.path.join(f"{train_dirname}/", "dqn.zip"))

            callback.save_history(f"{train_dirname}/")
            callback.save_best_model_weights(f"{train_dirname}/")

def test_flappy(dirpath, fps):
    test_env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar = False, background=None)
    test_env = FlappyBirdImgObservation(test_env)


    dqn = TrainerDQN.load_from_file(obs_space_shape=test_env.observation_space.shape,
                                    obs_space_dtype=test_env.observation_space.dtype,
                                    action_space_dim=test_env.action_space.n,
                                    filepath=os.path.join(dirpath, "dqn.zip"),
                                    model_class=FlappyBirdCNN,
                                    use_cuda_device=True,
                                    mode="eval")
    
    # load best model weights if possible
    best_model_path = os.path.join(dirpath, "eval_best_model.pth")
    if os.path.isfile(best_model_path):
        dqn.load_weights(best_model_path)
    
    SCALE = 6
    render_screen_size = (test_env.observation_space.shape[1]*SCALE, test_env.observation_space.shape[0]*SCALE)

    test_env = FrameStackObservation(test_env, stack_size=4, padding_type="zero")

    df_score = pd.DataFrame(columns=['score'])

    pygame.init()
    screen = pygame.display.set_mode(render_screen_size)
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 10*SCALE)

    scores = []

    running = True
    while running:
        obs, _ = test_env.reset()
        done = False
        truncated = False
        score = 0

        while not (done or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break

            action = dqn.predict(obs)
            obs, _, done, truncated, info = test_env.step(action)

            score = info['score']

            frame = test_env.render()

            frame = pygame.surfarray.make_surface(frame)
            frame = pygame.transform.rotate(frame, 90)
            frame = pygame.transform.flip(frame, False, True)
            frame = pygame.transform.scale(frame, screen.get_size())
            screen.blit(frame, (0,0))

            score_text_surf = font.render(f"score: {score}", True, (0, 0, 0))
            screen.blit(score_text_surf, 
                        (screen.get_size()[0]//2 - score_text_surf.get_size()[0]//2,
                        screen.get_size()[1]-score_text_surf.get_size()[1]*2))

            pygame.display.flip()
            clock.tick(fps)

        scores.append(score)
        df = pd.DataFrame({"score": scores})
        print(df.describe())

import argparse

parser = argparse.ArgumentParser(description="Program to train or evalute DQN for flappy bird enviroment.")
parser.add_argument('--mode', type=str, choices=["train", "test"], 
                    help="program mode", required=True)
parser.add_argument('--dirpath', type=str, help="path to directory to initialize from, leave empty to start train new model", required=False, default=None)
parser.add_argument('--hyperparams', type=str, help="path to hyperparameters file, (will override default)", required=False, default=None)
parser.add_argument('--test_fps', type=int, help="framerate of testing enviroment", required=False, default=30)
parser.add_argument('--train_steps', type=int, help="training duration in env steps", required=False, default=200_000)

args = parser.parse_args()

if args.mode == "train":
    train_flappy(dirpath = args.dirpath, hyperparams_path=args.hyperparams, steps=args.train_steps)
elif args.mode == "test":
    if args.dirpath is None:
        print("Error: must specify dirpath for testing.")
        sys.exit(1)
    test_flappy(dirpath = args.dirpath, fps=args.test_fps)
