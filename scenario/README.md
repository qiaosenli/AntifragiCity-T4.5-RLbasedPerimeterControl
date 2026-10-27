# Bundled experimental demand profiles

These route files are byte-for-byte copies of the inputs recorded in the three
training runs below. They are not regenerated demand. Run commands from the
`shared_green_rl_core` root. The network and default routes remain under
`LuSTScenario-master/scenario/`; this folder contains optional demand profiles.

| Profile folder | File | How to load | Total trips departing 08:00-09:00 |
|---|---|---|---:|
| `two_stage_strict_timestate` | `all_demand.rou.xml` | Replace original routes with `--route-files` | 30,382 |
| `ba_plus3000_secondhalf` | `extra_ba.rou.xml` | Append to original routes with `--extra-routes` | 28,382 |
| `aa_plus3000` | `extra_aa.rou.xml` | Append to original routes with `--extra-routes` | 28,382 |

Use one profile per run. The original baseline has 25,382 departures during
08:00-09:00. Each supplemental file contains only the 3,000 additional trips;
loading it as a complete replacement would remove the original trips.

## Two-stage strict

Original baseline plus 3,000 trips with controlled B-to-A crossings and 2,000
A-to-B trips. Departures are redistributed: trips with B-to-A crossings depart
in the second half, and no first-half planned route uses the specified entrance
edges. There are 14,489 first-half and 15,893 second-half departures. These are
planned-route/departure restrictions, not runtime closures or actual crossing
times. Many B-to-B trips visit A, so this is not an exclusive A-to-B/B-to-A
origin-destination split.

Historical training run: `two_stage_strict_timestate_reference_training`.
Original input: `data/demand_profiles/two_stage_strict_entries_ba3000_ab2000_08_09/all_demand.rou.xml`.

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 50 --generators 8 --begin 28800 --duration 3600 --state-scale 1000 --derivative-scale 100 --seed 42 --sumo-seed 42 --exploration-seed 10000 --noise 2 --route-files scenario/two_stage_strict_timestate/all_demand.rou.xml --output outputs/two_stage_strict_reference_training
```

## BA+3000 second-half

Keep the original baseline departures unchanged and add 3,000 trips whose
planned routes cross B-to-A through the selected controlled entrance edges.
All added departures are in 08:30-09:00, with a Gaussian-like peak at 08:40 and
a five-minute standard deviation, truncated to that half-hour. Eligible routes
can start and end in B while passing through A; the selection uses crossings.

Historical training run: `baseline_ba3000_secondhalf_time_reference_training`.
Original input: `data/demand_profiles/baseline_controlled_ba_plus3000_0830_09/extra_ba.rou.xml`.

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 50 --generators 8 --begin 28800 --duration 3600 --state-scale 1000 --derivative-scale 100 --seed 42 --sumo-seed 42 --exploration-seed 10000 --noise 2 --extra-routes scenario/ba_plus3000_secondhalf/extra_ba.rou.xml --output outputs/ba3000_secondhalf_reference_training
```

## AA+3000

Keep the original baseline unchanged and add 3,000 trips starting and ending
in A. The first 2,000 copy baseline A-to-A routes; another 1,000 improve spatial
coverage by using less-used A edges. Added departures follow a Gaussian-like
distribution centred at 08:30 with a ten-minute standard deviation, truncated
to 08:00-09:00. Some copied A-to-A routes pass through B en route.

Historical training run: `aa_plus3000_reference_training`.
Original input: `data/demand_profiles/aa_plus3000_spatial_gaussian_08_09/extra_aa.rou.xml`.

```sh
python -B -m shared_green_rl.train_parallel --training-mode reference --episodes 50 --generators 8 --begin 28800 --duration 3600 --state-scale 1000 --derivative-scale 100 --seed 42 --sumo-seed 42 --exploration-seed 10000 --noise 2 --extra-routes scenario/aa_plus3000/extra_aa.rou.xml --output outputs/aa3000_reference_training
```

The historical AA+3000 run used six state features, without elapsed time. This
command uses the current seven-feature core including elapsed time. The demand
is identical, but this is a fresh time-state experiment, not an exact rerun of
that historical training setup. Do not load its old six-input checkpoint into
the current core trainer.

All commands use the current reference-training defaults for the omitted
learning parameters (sample size 1,000, critic epochs 128, critic batch 256,
validation fraction 0.2, patience 20, two actor steps, target update every five
rounds) and `--min-green 5`. No training or simulation outputs are bundled.
