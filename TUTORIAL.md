# AntifragiCity T4.5 — Practical Tutorial

## Funding and disclaimer

AntifragiCity has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement **No. 101203052**.

The following disclaimer is adapted from Deliverable D4.5 for this documentation:

> The information, documentation and figures available in this documentation are provided by the AntifragiCity project's consortium under EC grant agreement 101203052 and do not necessarily reflect the views of the European Commission. The European Commission is not liable for any use that may be made of the information contained herein.

**Project copyright notice:** © ANTIFRAGICITY 2025-2030.
## How to use this tutorial

This guide covers setup, training and output interpretation for D4.5, *Operational Scenarios and Policies*. See the [README](README.md) for the project overview and the [scenario guide](scenario/README.md) for the three experimental demand profiles and their commands.

## 1. What this program does

One DDPG controller regulates 17 boundary traffic lights in the LuST network. Every 90 simulated seconds, it chooses a shared B-to-A green-time budget between 10 and 55 seconds. Region A is the protected area; Region B is the surrounding outskirt network. SUMO simulates traffic and Python controls the lights through TraCI.

Training maximizes completed trips. Total Time Spent (TTS) is logged separately. Each generator runs an independent SUMO simulation and collects experience for one shared learner.

## 2. Install and check the environment

Use Python 3.12 and SUMO 1.24.0. Install SUMO separately; `requirements.txt` installs NumPy and PyTorch. Run all commands from the repository root.

### Windows PowerShell

```powershell
cd "C:\path\to\shared_green_rl_core"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:SUMO_HOME = "C:\path\to\SUMO"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
```

Replace the paths with your installation locations. If you skip activation, use `.\.venv\Scripts\python.exe` in place of `python`.

### Linux

```bash
cd /path/to/shared_green_rl_core
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export SUMO_HOME=/path/to/sumo
export PATH="$SUMO_HOME/bin:$PATH"
```

`SUMO_HOME` must point to the installation containing `tools/traci`. Check both installations and the training entry point:

```sh
sumo --version
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
python -B -m shared_green_rl.train_parallel --help
```

## 3. Make a small first run

Run two generators for one 90-second cycle each, followed by a small learning update:

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 1 --generators 2 --duration 90 --train-sample-size 2 --critic-epochs 2 --critic-batch-size 2 --critic-patience 2 --output outputs/first_check
```

A successful run prints `generators: 2` and `transitions: 2`, and creates `episodes.csv`, `agent.pt` and generator folders under `outputs/first_check`. Every run requires a **new output directory**.

This checks the installation. Start a fresh run for the full experiment below.

## 4. Run the usual one-hour experiment

This command uses baseline demand, 50 training rounds and eight generators:

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 50 --generators 8 --train-sample-size 1000 --critic-epochs 128 --critic-batch-size 256 --critic-validation-fraction 0.2 --critic-patience 20 --actor-steps 2 --target-update-period 5 --noise 2 --seed 42 --sumo-seed 42 --exploration-seed 10000 --begin 28800 --duration 3600 --min-green 5 --state-scale 1000 --derivative-scale 100 --output outputs/reference_training
```

Keep `--training-mode reference` explicit; the CLI defaults to `legacy`. Each logged episode is one round:

1. Save the frozen policy as `episode_000/rollout_agent.pt`.
2. Run eight SUMO simulations of 08:00-09:00, collecting 40 transitions each.
3. Add the 320 transitions to shared replay and update the critic and actor.
4. Save the learner, replay and metrics, then begin the next round.

Fifty rounds run 400 simulations, with up to eight running at once. Adjust `--generators` to available CPU and memory; this also changes the number of transitions collected per round. The learner runs on CPU.

## 5. Choose demand and simulation inputs

Without route options, training uses the bundled baseline demand. The three experimental profiles follow the order in D4.5:

| Scenario | Profile | Option to add to the training command | Trips departing 08:00-09:00 |
|---|---|---|---:|
| 1 | AA+3000 | `--extra-routes scenario/aa_plus3000/extra_aa.rou.xml` | 28,382 |
| 2 | BA+3000 second-half | `--extra-routes scenario/ba_plus3000_secondhalf/extra_ba.rou.xml` | 28,382 |
| 3 | Two-stage strict | `--route-files scenario/two_stage_strict_timestate/all_demand.rou.xml` | 30,382 |

Use one profile per run and choose a distinct output folder. `--extra-routes` adds demand to the baseline; `--route-files` replaces it. Relative paths are resolved from the working directory. See the [scenario guide](scenario/README.md) for descriptions and complete commands.

Keep the bundled network, region definitions and additional files in their existing folder structure. SUMO settings come from `LuSTScenario-master/scenario/dua.static_1h_phase_separation.sumocfg`; `--begin`, `--duration` and `--sumo-seed` set the run's time window and seed. Each rollout writes a configuration copy and simulation outputs to its own output folder.

## 6. What the agent sees

The state contains seven normalized features in this order:

| Index | Feature | Default divisor |
|---|---|---:|
| 0 | Region A vehicle count, `n_A` | 1000 |
| 1 | Region B vehicle count, `n_B` | 1000 |
| 2 | Change in A count, `dn_A` | 100 |
| 3 | Change in B count, `dn_B` | 100 |
| 4 | Change in `dn_A`, `d2n_A` | 100 |
| 5 | Change in `dn_B`, `d2n_B` | 100 |
| 6 | Elapsed seconds | Episode duration, normally 3600 |

For control cycle `k`:

```text
dn[k]  = n[k] - n[k-1]
d2n[k] = dn[k] - dn[k-1]
```

These are differences per 90-second cycle, without division by seconds. Normalization divides by the listed values; it does not clip observations.

Counts follow road-edge membership. Vehicles on internal junction lanes use their incoming route edge. Vehicles without a resolvable external edge are excluded. SUMO starts with an empty network and no warm-up, so the first state and initial differences are zero.

## 7. What the agent outputs and how lights use it

The actor has dimensions `7 -> 64 -> 64 -> 1`, with ReLU hidden activations. Its `tanh` output is mapped to the allowed green interval. Exploration adds Gaussian noise with a default standard deviation of 2 seconds, then clips the action to the bounds.

At every light, the action sets total B-to-A green time. Clearance durations stay fixed, and A-to-B phases share the remaining green:

```text
A_to_B_green = 90 - clearance - B_to_A_green
```

`--min-green 5` applies to each individual green phase. The supplied network therefore permits a shared action from 10 to 55 seconds. Phase durations are allocated to two decimal places; SUMO advances in one-second steps. All 17 controlled programs restart at phase zero each cycle.

The direction mapping depends on the supplied network's phase order. Review `shared_green_rl/signals.py` before changing that network or its signal phases.

## 8. Reward, TTS and learning

```text
cycle_reward = completed_trips_during_cycle / 90
cycle_TTS    = sum(active_vehicle_count at each one-second step)
```

Completions are network-wide arrivals. TTS is measured in vehicle-seconds during the simulation window, including vehicles still running at the end. It excludes vehicles waiting to enter the network. For a one-hour rollout, total reward is `completed / 90`, while completion rate is `completed / 3600`.

The critic has dimensions `8 -> 64 -> 64 -> 1`: seven state values and one action, with the action scaled to `[-1, 1]`. Reference training:

1. Samples up to 1,000 transitions from a replay buffer of capacity 10,000.
2. Computes fixed Bellman targets using discount factor `gamma=0.99`.
3. Fits the critic with an 80/20 training/validation split for up to 128 epochs, stopping after 20 epochs without improvement. The final weights are retained.
4. Takes two actor steps using the sampled states.
5. Copies the online networks to their targets every five rounds.

## 9. Find and interpret outputs

```text
outputs/reference_training/
  settings.json             Run arguments and agent settings
  episodes.csv              Mean metrics per training round
  generators.csv            Metrics per generator and round
  agent.pt                  Learner saved after central updates
  replay_buffer.json        Replay data for resume
  episode_000/
    rollout_agent.pt        Frozen policy used for this round
    training.json           Losses and critic epoch history
    generator_000/
      episode.json          Rollout metrics
      samples.json          Collected transitions
      baseline.sumocfg      Generated SUMO configuration
      ...                   SUMO logs and outputs
```

| Metric | Meaning |
|---|---|
| `mean_tts` | Mean TTS across generators, in vehicle-seconds |
| `mean_completed` | Mean completed trips |
| `mean_completion_rate` | Mean completions per simulated second |
| `mean_reward` | Mean sum of cycle rewards |
| `transitions` | New transitions collected, normally 320 per round |
| `actor_loss`, `critic_loss` | Central learner losses |
| `collection_seconds`, `training_seconds`, `elapsed_seconds` | Collection, training and total wall time |

Each `samples.json` entry is `[state, action, reward, next_state, done]`. States are normalized using the divisors in `settings.json`.

Episode metrics describe the frozen `rollout_agent.pt` policy. The root `agent.pt` is saved after learning and overwritten each round. Training metrics include exploration noise; critic validation uses replay data. **`--noise 0` still trains the learner.**

## 10. Resume training

Keep the completed run, including `settings.json`, `episodes.csv`, `agent.pt` and `replay_buffer.json`, then write the continuation to a new folder:

```sh
python -B -m shared_green_rl.train_parallel --resume outputs/reference_training --episodes 50 --output outputs/reference_training_continued
```

`--episodes` specifies additional rounds. After rounds 0-49, this command runs rounds 50-99. The learner, replay, optimizers and random state are restored. Only `--episodes`, `--output` and `--resume` may differ; start a fresh experiment to change demand or other settings. Keep referenced demand files available.

## 11. Command-line parameter reference

The tables show parser defaults. The example command explicitly selects reference mode and 50 rounds; resumed runs inherit their saved settings.

| Argument | Default | Purpose |
|---|---|---|
| `--training-mode` | `legacy` | Set to `reference` for the procedure above |
| `--episodes` | 10 | Additional collection/training rounds |
| `--generators` | 8 | Parallel rollouts per round |
| `--begin` | 28800 | Start time in seconds after midnight (08:00) |
| `--duration` | 3600 | Positive episode duration divisible by 90 |
| `--noise` | 2 | Nonnegative exploration standard deviation in seconds |
| `--seed` | 42 | Learner initialization and replay sampling seed |
| `--sumo-seed` | 42 | SUMO seed shared by all rollouts |
| `--exploration-seed` | 10000 | Base seed for generator action noise |
| `--state-scale` | 1000 | Positive divisor for regional counts |
| `--derivative-scale` | 100 | Positive divisor for first and second differences |
| `--min-green` | 5 | Minimum seconds per green phase, rounded upward |
| `--sumo` | `sumo` | SUMO executable name or path |
| `--extra-routes` | None | Append supplemental demand |
| `--route-files` | None | Replace demand; cannot combine with `--extra-routes` |
| `--output` | Timestamped folder under `shared_green_rl/outputs/` | New output directory |
| `--resume` | None | Completed run to continue |

Reference learning parameters:

| Argument | Default | Purpose |
|---|---:|---|
| `--train-sample-size` | 1000 | Maximum transitions sampled per round; at least 2 |
| `--critic-epochs` | 128 | Maximum critic epochs |
| `--critic-batch-size` | 256 | Critic minibatch size |
| `--critic-validation-fraction` | 0.2 | Validation fraction, strictly between 0 and 1 |
| `--critic-patience` | 20 | Epochs without validation improvement before stopping |
| `--actor-steps` | 2 | Actor updates per round |
| `--target-update-period` | 5 | Rounds between target-network copies |

The remaining CLI options, `--batch-size` (32) and `--updates` (one per new transition by default), apply to legacy mode. Omit `--updates` in reference mode.

Generator `g` in round `e` uses action-noise seed `exploration_seed + e * generators + g`. Learning rates are defined in `ddpg_agent.py`: `actor_lr=1e-4` and `critic_lr=1e-3`.

## 12. File-by-file reading guide

Read `train_parallel.py` for orchestration, `generating.py` for rollout workers, `environment.py` for state and reward, `signals.py` for green allocation, and `ddpg_agent.py` for learning. `sumo_runtime.py` handles SUMO inputs and regional counts. See the [repository guide](README.md#repository-guide) for links.

## 13. Common problems

| Symptom | Action |
|---|---|
| Missing NumPy or PyTorch | Activate the virtual environment and install `requirements.txt` with that interpreter. |
| `No module named shared_green_rl` | Run the `python -m shared_green_rl.train_parallel` command from the repository root. |
| SUMO or TraCI not found | Check `sumo --version`, PATH and `SUMO_HOME/tools/traci`. |
| Output directory already exists | Choose a new output folder, including when resuming. |
| Missing route or additional file | Check demand paths and retain the bundled scenario folder structure. |
| Resume rejects settings | Retain the saved settings or start a fresh experiment. |
| No new episode row yet | A row is written after all generators and the central update finish. |
| Loss changes without better traffic performance | Assess completed trips and TTS alongside learner losses. |
