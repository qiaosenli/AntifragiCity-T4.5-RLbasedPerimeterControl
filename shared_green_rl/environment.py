"""Six traffic features plus episode time, scalar green budget and completion reward."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from shared_green_rl.sumo_runtime import SCENARIO, load_regions, prepare_config, load_traci, measure_accumulation
from shared_green_rl.signals import NETWORK, SignalMapping


class Environment:
    control_interval = 90
    features = ['n_A', 'n_B', 'dn_A', 'dn_B', 'd2n_A', 'd2n_B', 'elapsed_fraction']

    def __init__(self, begin=28800, duration=3600, min_green=5, scale=1000, sumo='sumo', extra_routes=None,
                 derivative_scale=100, route_files=None):
        self.mapping = SignalMapping(min_green)
        self.begin, self.duration = begin, duration
        self.scale = np.array([scale, scale, derivative_scale, derivative_scale,
                               derivative_scale, derivative_scale, duration], dtype=np.float32)
        self.extra_routes = Path(extra_routes).resolve() if extra_routes else None
        self.route_files = Path(route_files).resolve() if route_files else None
        if self.extra_routes and self.route_files:
            raise ValueError('Use either replacement routes or supplemental routes')
        if duration <= 0 or duration % 90 or np.any(self.scale <= 0):
            raise ValueError('Use positive duration divisible by 90 and positive scale')
        self.binary = shutil.which(sumo)
        if not self.binary:
            raise FileNotFoundError(sumo)
        self.traci = load_traci(self.binary)
        self.running = False

    def reset(self, output, seed):
        self.close()
        output = Path(output).resolve()
        if output.is_relative_to(SCENARIO.resolve()):
            raise ValueError('Keep outputs outside the scenario')
        output.mkdir(parents=True, exist_ok=False)
        source = SCENARIO / 'dua.static_1h_phase_separation.sumocfg'
        cfg = ET.parse(source)
        cfg.find('input/net-file').set('value', str(NETWORK))
        if self.route_files:
            cfg.find('input/route-files').set('value', str(self.route_files))
        if self.extra_routes:
            routes = cfg.find('input/route-files')
            routes.set('value', routes.get('value') + ',' + str(self.extra_routes))
        generated, _ = prepare_config(cfg, source, output)
        # Preserve other lights; prevent the old additional file overriding these 17.
        for name in cfg.find('input/additional-files').get('value').split(','):
            tree = ET.parse(name)
            # Edited lanes invalidate some legacy detector positions. Copies only.
            for detector in tree.iter('e1Detector'):
                detector.set('friendlyPos', 'true')
            for light in list(tree.getroot().findall('tlLogic')):
                if light.get('id') in {s.tls_id for s in self.mapping.signals}:
                    tree.getroot().remove(light)
            tree.write(name, encoding='utf-8', xml_declaration=True)
        self.edges, self.region_a = load_regions(NETWORK, SCENARIO / 'Region_A.txt')
        self.traci.start([self.binary, '-c', str(generated), '--begin', str(self.begin),
            '--end', str(self.begin+self.duration), '--seed', str(seed), '--no-step-log', 'true'])
        self.running = True
        if self.traci.simulation.getDeltaT() != 1:
            raise ValueError('Expected one-second SUMO steps')
        self.n = self.accumulation()
        self.dn = np.zeros(2)
        return np.concatenate((self.n, self.dn, self.dn, [0.])).astype(np.float32) / self.scale

    def accumulation(self):
        # Only once per 90 seconds, never in the per-second loop. No flow features.
        counts = measure_accumulation(self.traci, self.edges, self.region_a)
        return np.array([counts['A'], counts['B']], dtype=float)

    def step(self, action):
        budget, plans = self.mapping.plan(float(action[0]))
        tls = self.traci.trafficlight
        for s in self.mapping.signals:
            phases = [tls.Phase(d, state) for d, state in zip(plans[s.tls_id], s.states)]
            tls.setProgramLogic(s.tls_id, tls.Logic('shared_green', 0, 0, phases))
            tls.setProgram(s.tls_id, 'shared_green')
            tls.setPhase(s.tls_id, 0)
            tls.setPhaseDuration(s.tls_id, phases[0].duration)
        tts = completed = 0
        for _ in range(self.control_interval):
            self.traci.simulationStep()
            tts += self.traci.vehicle.getIDCount()
            completed += self.traci.simulation.getArrivedNumber()
        n = self.accumulation()
        dn = n - self.n
        elapsed = self.traci.simulation.getTime() - self.begin
        state = np.concatenate((n, dn, dn-self.dn, [elapsed])).astype(np.float32) / self.scale
        self.n, self.dn = n, dn
        done = self.traci.simulation.getTime() >= self.begin + self.duration
        return state, completed / self.control_interval, done, dict(
            tts=tts, completed=completed, completion_rate=completed / self.control_interval,
            ba_seconds=budget)

    def close(self):
        if self.running:
            self.traci.close()
            self.running = False
