# AntifragiCity T4.5 — Operational Scenarios and Policies

## Funding and disclaimer

AntifragiCity has received funding from the European Union's Horizon Europe
research and innovation programme under grant agreement **No. 101203052**.

The following disclaimer is adapted from Deliverable D4.5 for this documentation:

> The information, documentation and figures available in this documentation
> are provided by the AntifragiCity project's consortium under EC grant
> agreement 101203052 and do not necessarily reflect the views of the European
> Commission. The European Commission is not liable for any use that may be
> made of the information contained herein.

**Project copyright notice:** © ANTIFRAGICITY 2025-2030.

## Overview

This repository contains the standalone training core for the AntifragiCity
T4.5 microscopic perimeter-control study. A Deep Deterministic Policy Gradient
(DDPG) controller interacts with Simulation of Urban MObility (SUMO) through
TraCI in the Luxembourg SUMO Traffic (LuST) scenario. Every 90 simulated
seconds, it selects one shared inward (B-to-A) green-time budget for 17 boundary
traffic lights between a protected Region A and the surrounding Region B.

D4.5 builds on the microscopic simulation work of T4.3 and the macroscopic
perimeter-control methodology of T4.4. This implementation learns from regional
vehicle counts, their first and second differences, and elapsed time. Its reward
is completed trips per control interval; total time spent (TTS) is recorded
separately. The reward has no additional antifragility or redundancy term.

The package includes the learner, parallel SUMO rollout workers, required
scenario inputs and three experimental demand profiles. The report also covers
fixed-time comparisons and performance analysis; this package does not include
those evaluation tools, plotting scripts, demand-generation scripts, saved
results or trained checkpoints. Training creates new metrics and checkpoints.
The current seven-input state also differs from historical six-input runs, so
using their demand files does not by itself reproduce their reported results.

Start with the [practical tutorial](TUTORIAL.md) for a small installation check,
the control and learning workflow, output interpretation and all parameters.
Use the [demand-profile guide](scenario/README.md) for profile provenance and
complete experiment commands.

## Installation

Run commands from this repository's root directory. Python **3.12** and SUMO
**1.24.0** were used for the existing experiments. The Python requirements pin
NumPy and PyTorch; the learner runs on CPU.

Install SUMO separately and ensure `sumo` is on PATH. Set `SUMO_HOME` to the SUMO
installation directory so `$SUMO_HOME/tools/traci` is available.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
sumo --version
```

## Train and resume

The command below uses the original baseline demand, eight generators and the
reference training procedure. Keep `--training-mode reference` explicit: the
preserved CLI also supports legacy training and defaults to that mode.

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 50 --generators 8 --train-sample-size 1000 --critic-epochs 128 --critic-batch-size 256 --critic-validation-fraction 0.2 --critic-patience 20 --actor-steps 2 --target-update-period 5 --noise 2 --seed 42 --sumo-seed 42 --exploration-seed 10000 --begin 28800 --duration 3600 --min-green 5 --state-scale 1000 --derivative-scale 100 --output outputs/reference_training
```

Each round collects eight independent one-hour SUMO rollouts using one frozen
policy, then updates the shared learner. Fifty rounds therefore run 400
simulations. Choose a **new output directory** for every command; existing
directories are rejected. For a shorter first run, follow
[Tutorial §3](TUTORIAL.md#3-make-a-small-first-run).

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
of additional rounds. Other experiment settings are inherited and must remain
unchanged.

```sh
python -B -m shared_green_rl.train_parallel --resume outputs/reference_training --episodes 50 --output outputs/reference_training_continued
```

## Control and learning at a glance

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

## Outputs and interpretation

`episodes.csv` contains generator-averaged metrics; `generators.csv` records
each individual rollout. Each episode's
`rollout_agent.pt` is the frozen policy that produced those rollouts; the root
`agent.pt` is saved after learning. Resume also requires `replay_buffer.json`
and `settings.json`, together with the completed run's `episodes.csv`.
Outputs should remain outside the scenario directory.

Training metrics include exploration noise. Critic validation uses replay
transitions and is not a separate traffic-policy evaluation. Setting
`--noise 0` still trains the learner. See
[Tutorial §9](TUTORIAL.md#9-find-and-interpret-outputs) for the output layout
and metric definitions.

## Repository guide

| Path | Purpose |
|---|---|
| [TUTORIAL.md](TUTORIAL.md) | Setup, first run, methodology, parameters and troubleshooting |
| [ddpg_agent.py](ddpg_agent.py) | Actor, critic, replay, learning updates and checkpoints |
| [shared_green_rl/train_parallel.py](shared_green_rl/train_parallel.py) | Training entry point: parallel collection and central learning |
| [shared_green_rl/generating.py](shared_green_rl/generating.py) | One frozen-policy SUMO rollout per generator |
| [shared_green_rl/environment.py](shared_green_rl/environment.py) | Observations, completion reward and control-cycle stepping |
| [shared_green_rl/signals.py](shared_green_rl/signals.py) | Shared green-time allocation across the 17 lights |
| [shared_green_rl/sumo_runtime.py](shared_green_rl/sumo_runtime.py) | Configuration copies, SUMO access and regional accumulation |
| [LuSTScenario-master/](LuSTScenario-master/) | Required network, original demand, region definitions and upstream licence |
| [scenario/](scenario/README.md) | Three experimental demand profiles and selection commands |

The training loop, rollout worker, signal allocation and agent learning methods
are preserved from `shared_green_rl_server`. Packaging removes the agent's
self-test and replaces the baseline/audit dependency with runtime-only helpers.
Vehicles on internal junction lanes are assigned to regions using the incoming
route edge; vehicles with no road are excluded. No boundary audit reports are
generated.
