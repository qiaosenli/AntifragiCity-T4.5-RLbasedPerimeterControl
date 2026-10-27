"""Reference-style parallel complete rollouts, followed by centralized training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from ddpg_agent import DDPG
from shared_green_rl.environment import Environment
from shared_green_rl.generating import generating_process
from shared_green_rl.signals import SignalMapping
from shared_green_rl.sumo_runtime import write_json


REFERENCE_FIELDS = [
    'replay_buffer_size', 'fixed_sample_size', 'critic_train_size', 'critic_validation_size',
    'critic_epochs_completed', 'critic_optimizer_steps', 'actor_optimizer_steps',
    'critic_training_loss', 'critic_validation_loss', 'best_critic_validation_loss',
    'actor_losses', 'actor_loss_final', 'actor_loss_mean', 'mean_predicted_q',
    'mean_target_value', 'max_abs_predicted_q', 'max_abs_target_value',
    'hard_target_update', 'reference_rounds_completed']


def restore_training(folder):
    """Restore the learner, replay and next round from a completed run."""
    agent = DDPG.load(folder / 'agent.pt')
    for state, action, reward, next_state, done in json.loads(
            (folder / 'replay_buffer.json').read_text()):
        agent.remember(state, action, reward, next_state, done)
    with (folder / 'episodes.csv').open(newline='') as f:
        last = list(csv.DictReader(f))[-1]
    return agent, int(last['episode']) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generators', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=10, help='Additional collect/train rounds')
    parser.add_argument('--resume', type=Path, help='Completed run folder; restore learner and replay')
    parser.add_argument('--begin', type=int, default=28800)
    parser.add_argument('--duration', type=int, default=3600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--updates', type=int, default=None,
                        help='Legacy only: updates per round; default one per new transition')
    parser.add_argument('--training-mode', choices=['legacy', 'reference'], default='legacy')
    parser.add_argument('--train-sample-size', type=int, default=1000)
    parser.add_argument('--critic-epochs', type=int, default=128)
    parser.add_argument('--critic-batch-size', type=int, default=256)
    parser.add_argument('--critic-validation-fraction', type=float, default=0.2)
    parser.add_argument('--critic-patience', type=int, default=20)
    parser.add_argument('--actor-steps', type=int, default=2)
    parser.add_argument('--target-update-period', type=int, default=5)
    parser.add_argument('--noise', type=float, default=2, help='Exploration std in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Learner initialization and replay seed')
    parser.add_argument('--sumo-seed', type=int, default=42, help='Fixed SUMO seed for every rollout')
    parser.add_argument('--exploration-seed', type=int, default=10000, help='Base seed for independent rollout noise')
    parser.add_argument('--min-green', type=float, default=5)
    parser.add_argument('--state-scale', type=float, default=1000, help='Accumulation divisor')
    parser.add_argument('--derivative-scale', type=float, default=100, help='First/second difference divisor')
    parser.add_argument('--sumo', default='sumo')
    parser.add_argument('--extra-routes', help='Supplemental route XML appended to the original demand')
    parser.add_argument('--route-files', help='Complete replacement demand XML; do not combine with --extra-routes')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs' /
                        datetime.now().strftime('parallel_%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    if args.resume:
        previous = json.loads((args.resume / 'settings.json').read_text())
        if previous.get('state_features') != Environment.features:
            parser.error('Start fresh: checkpoint states do not include the current elapsed-time feature')
        if previous.get('green_duration_decimals', 0) != 2:
            parser.error('Start fresh: this replay used whole-second signal allocation')
        if previous['reward'] != 'completed_trips / control_interval':
            parser.error('Resume requires a completion-reward checkpoint')
        # Inherit the original experiment, while allowing a new duration in rounds/output.
        inherited = {k: v for k, v in previous['arguments'].items()
                     if k not in ('episodes', 'output', 'resume')}
        inherited.setdefault('extra_routes', None)
        inherited.setdefault('route_files', None)
        inherited.setdefault('training_mode', 'legacy')
        # Older checkpoints/replay used the accumulation divisor for every feature.
        inherited.setdefault('derivative_scale', inherited['state_scale'])
        parser.set_defaults(**inherited)
        args = parser.parse_args()
        for key, value in inherited.items():
            if getattr(args, key) != value:
                parser.error(f'Resume keeps --{key.replace("_", "-")}={value}')
    if args.route_files and args.extra_routes:
        parser.error('Use either --route-files or --extra-routes')
    if min(args.generators, args.episodes, args.batch_size) < 1 or (args.updates is not None and args.updates < 1):
        parser.error('Generators, episodes, batch size and updates must be positive')
    if args.training_mode == 'reference':
        if args.updates is not None:
            parser.error('--updates is legacy-only; omit it in reference mode')
        if min(args.train_sample_size, args.critic_epochs, args.critic_batch_size,
               args.critic_patience, args.actor_steps, args.target_update_period) < 1:
            parser.error('Reference training counts must be positive')
        if not 0 < args.critic_validation_fraction < 1:
            parser.error('Use --critic-validation-fraction between 0 and 1')
        if args.train_sample_size < 2:
            parser.error('Reference mode needs at least two samples for a validation split')
    if args.duration <= 0 or args.duration % 90 or min(args.state_scale, args.derivative_scale) <= 0 or args.noise < 0:
        parser.error('Use duration divisible by 90, positive scale and nonnegative noise')
    # torch.set_num_threads(1)
    output = args.output.resolve()
    mapping = SignalMapping(min_green=args.min_green)
    if args.resume:
        agent, start_episode = restore_training(args.resume)
    else:
        agent = DDPG(len(Environment.features), action_low=[mapping.low], action_high=[mapping.high],
                     batch_size=args.batch_size, seed=args.seed, normalize_critic_actions=True)
        start_episode = 0
    output.mkdir(parents=True, exist_ok=False)
    env_options = dict(duration=args.duration, begin=args.begin, scale=args.state_scale,
                       min_green=args.min_green, sumo=args.sumo, extra_routes=args.extra_routes,
                       derivative_scale=args.derivative_scale, route_files=args.route_files)
    write_json(output / 'settings.json', dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        start_episode=start_episode,
        green_duration_decimals=2,
        state_divisors=[args.state_scale]*2 + [args.derivative_scale]*4 + [args.duration],
        agent=agent.config, state_features=Environment.features,
        action_bounds=[mapping.low, mapping.high], control_interval=90,
        action='shared total B-to-A green seconds per light',
        reward='completed_trips / control_interval', reward_units='vehicles/second', critic_action_normalization='[-1, 1]',
        tts_units='vehicle-seconds',
        workflow='frozen-policy parallel rollouts then central replay updates'))
    fields = ['episode', 'generators', 'transitions', 'mean_reward', 'mean_tts', 'mean_completed', 'mean_completion_rate',
              'updates', 'actor_loss', 'critic_loss', 'collection_seconds',
              'training_seconds', 'elapsed_seconds', 'training_mode']
    if args.training_mode == 'reference':
        fields += REFERENCE_FIELDS
    # Spawn gives each worker its own TraCI module/connection on Windows and Linux.
    with ProcessPoolExecutor(max_workers=args.generators, mp_context=mp.get_context('spawn')) as pool, \
            (output / 'episodes.csv').open('w', newline='') as f, \
            (output / 'generators.csv').open('w', newline='') as gf:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        generator_writer = csv.DictWriter(gf, fieldnames=['episode', 'generator', 'sumo_seed', 'exploration_seed',
            'reward', 'tts', 'completed', 'completion_rate', 'transitions', 'elapsed_seconds'])
        generator_writer.writeheader()
        for episode in range(start_episode, start_episode + args.episodes):
            start = perf_counter()
            folder = output / f'episode_{episode:03d}'
            folder.mkdir()
            checkpoint = folder / 'rollout_agent.pt'
            agent.save(checkpoint)
            futures = [pool.submit(generating_process, checkpoint,
                        folder / f'generator_{g:03d}', env_options,
                        args.sumo_seed, args.exploration_seed + episode * args.generators + g, args.noise)
                       for g in range(args.generators)]
            # Blocking result collection; no progress polling or output inspection.
            stats = [future.result() for future in futures]
            collection_seconds = perf_counter() - start
            train_start = perf_counter()
            for g, stat in enumerate(stats):
                generator_writer.writerow(dict(episode=episode, generator=g, **stat))
                samples = json.loads((folder / f'generator_{g:03d}' / 'samples.json').read_text())
                for state, action, reward, next_state, done in samples:
                    agent.remember(np.asarray(state, dtype=np.float32),
                        np.asarray(action, dtype=np.float32), reward,
                        np.asarray(next_state, dtype=np.float32), done)
            transitions = sum(s['transitions'] for s in stats)
            losses, updates = None, 0
            reference_stats = {}
            if args.training_mode == 'reference':
                losses = agent.train_reference_round(
                    train_sample_size=args.train_sample_size, critic_epochs=args.critic_epochs,
                    critic_batch_size=args.critic_batch_size,
                    critic_validation_fraction=args.critic_validation_fraction,
                    critic_patience=args.critic_patience, actor_steps=args.actor_steps,
                    target_update_period=args.target_update_period)
                updates = ''  # Reference mode logs separate critic/actor step counts.
                if losses is not None:
                    write_json(folder / 'training.json', losses)
                    reference_stats = {key: losses[key] for key in REFERENCE_FIELDS}
                    reference_stats['actor_losses'] = json.dumps(losses['actor_losses'])
                    losses = {key: losses[key] for key in ('actor_loss', 'critic_loss')}
            else:
                for _ in range(args.updates if args.updates is not None else transitions):
                    losses = agent.train_step()
                    if losses is None:
                        break
                    updates += 1
            agent.save(output / 'agent.pt')
            write_json(output / 'replay_buffer.json', [
                [s.tolist(), a.tolist(), r, ns.tolist(), d]
                for s, a, r, ns, d in agent.memory])
            row = dict(episode=episode, generators=args.generators, transitions=transitions,
                       training_mode=args.training_mode, **reference_stats,
                       mean_reward=float(np.mean([s['reward'] for s in stats])),
                       mean_tts=float(np.mean([s['tts'] for s in stats])),
                       mean_completed=float(np.mean([s['completed'] for s in stats])),
                       mean_completion_rate=float(np.mean([s['completion_rate'] for s in stats])), updates=updates,
                       **(losses or dict(actor_loss='', critic_loss='')),
                       collection_seconds=collection_seconds,
                       training_seconds=perf_counter()-train_start,
                       elapsed_seconds=perf_counter()-start)
            writer.writerow(row)
            f.flush()
            gf.flush()
            print(row, flush=True)


if __name__ == '__main__':
    main()
