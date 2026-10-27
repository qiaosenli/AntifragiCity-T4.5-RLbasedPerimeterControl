"""Read the edited network and allocate one shared green budget."""
from dataclasses import dataclass
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

SCENARIO = Path(__file__).resolve().parent.parent / "LuSTScenario-master/scenario"
NETWORK = SCENARIO / 'lust_phase_separation.net.xml'
SPECIAL = {'-4214', '-8782', '-29900', '-17072', '-12'}


@dataclass
class Signal:
    tls_id: str
    states: list[str]
    durations: list[float]
    ba: list[int]
    ab: list[int]


class SignalMapping:
    def __init__(self, min_green=5):
        self.min_green = math.ceil(min_green)
        ids = json.loads((SCENARIO / 'boundary_intersections.json').read_text())['intersections']
        net = ET.parse(NETWORK)
        self.signals = []
        tll = net.findall('tlLogic')
        for tls in ids:
            for t in tll:
                if t.get('id') == tls:
                    program = t
                    break
            phases = program.findall('phase')
            states = [p.get('state') for p in phases]
            durations = [float(p.get('duration')) for p in phases]
            green_idx = [i for i, state in enumerate(states) if 'G' in state]
            split = 2 if tls in SPECIAL else 1
            self.signals.append(Signal(tls, states, durations, green_idx[:split], green_idx[split:]))
        # Shared B-to-A budget bounds preserve every phase's minimum green.
        self.low = max(len(s.ba) * self.min_green for s in self.signals)
        self.high = min(self.green_budget(s) - len(s.ab) * self.min_green for s in self.signals)
        if self.min_green <= 0 or self.low > self.high:
            raise ValueError('Infeasible minimum green')

    @staticmethod
    def green_budget(signal):
        return 90 - sum(d for i, d in enumerate(signal.durations)
                        if i not in signal.ba + signal.ab)

    def plan(self, seconds):
        if not math.isfinite(seconds) or not self.low <= seconds <= self.high:
            raise ValueError(f'B-to-A budget must be within [{self.low}, {self.high}]')
        budget = round(seconds, 2)
        plans = {}
        for s in self.signals:
            durations = s.durations.copy()
            for indices, total in ((s.ba, budget), (s.ab, self.green_budget(s) - budget)):
                # Allocate integer hundredths so the phase sum stays exact.
                base, extra = divmod(round(total * 100), len(indices))
                for j, index in enumerate(indices):
                    durations[index] = (base + (j < extra)) / 100
            plans[s.tls_id] = durations
        return budget, plans
