# Experimental demand profiles

The profiles below follow the scenario order in D4.5. Run commands from the repository root and choose a new output folder for each run. See the [README](../README.md) for project information and the [tutorial](../TUTORIAL.md) for setup and training details.

| Scenario | Profile | Demand file | How to load | Trips departing 08:00-09:00 |
|---|---|---|---|---:|
| 1 | AA+3000 | `aa_plus3000/extra_aa.rou.xml` | Append with `--extra-routes` | 28,382 |
| 2 | BA+3000 second-half | `ba_plus3000_secondhalf/extra_ba.rou.xml` | Append with `--extra-routes` | 28,382 |
| 3 | Two-stage strict | `two_stage_strict_timestate/all_demand.rou.xml` | Replace with `--route-files` | 30,382 |

Use one profile per run. The baseline has 25,382 departures during 08:00-09:00. Scenarios 1 and 2 each add 3,000 trips; Scenario 3 supplies the complete demand.

## Scenario 1: AA+3000

Adds 3,000 trips starting and ending in Region A. The first 2,000 copy baseline A-to-A routes; another 1,000 improve coverage of less-used A edges. Departures follow a Gaussian-like distribution centred at 08:30 with a ten-minute standard deviation, truncated to 08:00-09:00. Some routes pass through B.

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 50 --generators 8 --begin 28800 --duration 3600 --state-scale 1000 --derivative-scale 100 --seed 42 --sumo-seed 42 --exploration-seed 10000 --noise 2 --extra-routes scenario/aa_plus3000/extra_aa.rou.xml --output outputs/aa3000_reference_training
```

## Scenario 2: BA+3000 second-half

Adds 3,000 trips whose planned routes cross B-to-A through the controlled entrances. Added departures fall within 08:30-09:00, with a Gaussian-like peak at 08:40 and a five-minute standard deviation. Selection uses boundary crossings; trips can start and end in B while passing through A.

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 50 --generators 8 --begin 28800 --duration 3600 --state-scale 1000 --derivative-scale 100 --seed 42 --sumo-seed 42 --exploration-seed 10000 --noise 2 --extra-routes scenario/ba_plus3000_secondhalf/extra_ba.rou.xml --output outputs/ba3000_secondhalf_reference_training
```

## Scenario 3: Two-stage strict

Includes baseline demand plus 3,000 trips with controlled B-to-A crossings and 2,000 A-to-B trips. Departures are redistributed so trips with B-to-A crossings depart in the second half and no first-half planned route uses the selected entrance edges. There are 14,489 first-half and 15,893 second-half departures. These restrictions apply to planned routes and departure times.

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 50 --generators 8 --begin 28800 --duration 3600 --state-scale 1000 --derivative-scale 100 --seed 42 --sumo-seed 42 --exploration-seed 10000 --noise 2 --route-files scenario/two_stage_strict_timestate/all_demand.rou.xml --output outputs/two_stage_strict_reference_training
```

All commands use the reference-training defaults: sample size 1,000, up to 128 critic epochs, batch size 256, validation fraction 0.2, patience 20, two actor steps, target updates every five rounds and `--min-green 5`.
