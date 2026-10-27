# Shared-green RL core

New to the codebase? Start with [the practical tutorial](TUTORIAL.md) for setup,
first-run commands, module explanations, output interpretation and all tunable
parameters.

Minimal standalone package for parallel DDPG training with elapsed time in the
state. Run commands from this directory. Python 3.12 and SUMO 1.24.0 were used
for the existing experiments.

## Contents

- `ddpg_agent.py`: actor, critic, replay, training updates and checkpoints.
- `shared_green_rl/train_parallel.py`: parallel collection and central training.
- `shared_green_rl/generating.py`: one frozen-policy SUMO rollout per generator.
- `shared_green_rl/environment.py`: states, completion reward and cycle stepping.
- `shared_green_rl/signals.py`: shared green allocation across the 17 lights.
- `shared_green_rl/sumo_runtime.py`: configuration copies and accumulation counts.
- `LuSTScenario-master/`: only the default configuration's required inputs,
  region definitions, boundary-light IDs and upstream license.
- `scenario/`: the three experimental demand profiles used in training;
  see [profile selection and commands](scenario/README.md).

There are no plotting, testing, summarizing, evaluation or demand-generation
scripts, saved outputs or checkpoints in this package.
Training still writes its normal metrics and checkpoints when run.

The training loop, rollout worker, signal allocation and agent learning methods
are preserved from `shared_green_rl_server`. Packaging removes the agent's
self-test and replaces the baseline/audit dependency with runtime-only helpers.
Accumulation keeps the original rule: vehicles on internal junction lanes are
assigned using the incoming route edge; vehicles with no road are excluded.
No boundary audit reports are generated.

## Setup

Install SUMO separately and ensure `sumo` is on PATH. Set `SUMO_HOME` to the SUMO
installation directory so `$SUMO_HOME/tools/traci` is available.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
sumo --version
```

## Train

The command below uses the original baseline demand, eight generators and the
reference training procedure. Keep `--training-mode reference` explicit: the
preserved CLI also supports legacy training and defaults to that mode.

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 50 --generators 8 --train-sample-size 1000 --critic-epochs 128 --critic-batch-size 256 --critic-validation-fraction 0.2 --critic-patience 20 --actor-steps 2 --target-update-period 5 --noise 2 --seed 42 --sumo-seed 42 --exploration-seed 10000 --begin 28800 --duration 3600 --min-green 5 --state-scale 1000 --derivative-scale 100 --output outputs/reference_training
```

To use a bundled demand profile, add one of these options to the training
command and choose a distinct output directory:

| Profile | Demand option |
|---|---|
| Two-stage strict (time-state experiment) | `--route-files scenario/two_stage_strict_timestate/all_demand.rou.xml` |
| BA+3000 second-half | `--extra-routes scenario/ba_plus3000_secondhalf/extra_ba.rou.xml` |
| AA+3000 | `--extra-routes scenario/aa_plus3000/extra_aa.rou.xml` |

Use only one profile at a time. The first replaces the original demand; the
other two append 3,000 trips to it. Relative paths are resolved from your working
directory. See [scenario/README.md](scenario/README.md) for complete commands.

Resume a completed run into a new output directory; `--episodes` is the number
of additional rounds. Other experiment settings are inherited.

```sh
python -B -m shared_green_rl.train_parallel --resume outputs/reference_training --episodes 50 --output outputs/reference_training_continued
```

## State, action and training

The seven inputs, in order, are:

```text
[n_A/1000, n_B/1000, dn_A/100, dn_B/100, d2n_A/100, d2n_B/100, elapsed_seconds/duration]
```

Accumulation is sampled once per 90-second control cycle. `dn` is the difference
from the previous cycle, and `d2n` is the difference between consecutive `dn`
values; neither is divided by seconds. Initial differences are zero and time
runs from zero to one. Six-input historical checkpoints are incompatible.

The scalar action is a shared B-to-A green budget, normally 10 to 55 seconds.
The cycle is 90 seconds. Clearance durations stay fixed; remaining green is
allocated to A-to-B phases. Multiple phases in each direction share its budget,
with durations allocated to two decimal places. SUMO uses one-second steps.

Reward is completed trips divided by 90 seconds. TTS remains a logged metric,
not a reward term. Every generator uses SUMO seed 42 by default; exploration
uses independent seeds. A full one-hour rollout provides 40 transitions per
generator, or 320 per eight-generator round.

Reference training samples up to 1,000 replay transitions, freezes their target
values, fits the critic with an 80/20 train/validation split and patience-based
stopping, takes two actor steps, and updates target networks every five rounds.

`episodes.csv` contains generator-averaged metrics. Each episode's
`rollout_agent.pt` is the frozen policy that produced those rollouts; the root
`agent.pt` is saved after learning. Resume also requires `replay_buffer.json`
and `settings.json`. Outputs should remain outside the scenario directory.
