"""One isolated rollout generator: frozen policy, independent SUMO and noise."""
from time import perf_counter
import numpy as np
import torch
from ddpg_agent import DDPG
from shared_green_rl.environment import Environment
from shared_green_rl.sumo_runtime import write_json


def generating_process(checkpoint, folder, env_options, sumo_seed, exploration_seed, noise_std):
    start = perf_counter()
    torch.set_num_threads(1)
    agent = DDPG.load(checkpoint)
    agent.rng = np.random.default_rng(exploration_seed)
    env = Environment(**env_options)
    samples, total_reward, total_tts, total_completed = [], 0.0, 0.0, 0
    try:
        state = env.reset(folder, sumo_seed)
        done = False
        while not done:
            action = agent.choose_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)
            samples.append([state.tolist(), action.tolist(), reward, next_state.tolist(), done])
            total_reward += reward
            total_tts += info['tts']
            total_completed += info['completed']
            state = next_state
    finally:
        env.close()
    write_json(folder / 'samples.json', samples)
    stats = dict(sumo_seed=sumo_seed, exploration_seed=exploration_seed, reward=total_reward, tts=total_tts,
                 completed=total_completed, completion_rate=total_completed / env.duration,
                 transitions=len(samples), elapsed_seconds=perf_counter() - start)
    write_json(folder / 'episode.json', stats)
    return stats
