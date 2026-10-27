"""SUMO configuration, regional accumulation and JSON output used by training."""
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

SCENARIO = Path(__file__).resolve().parents[1] / 'LuSTScenario-master/scenario'

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def prepare_config(cfg, source, output):
    """Copy configuration and detector definitions; never write into the scenario."""
    inputs = [source]
    for element in cfg.findall('input/*'):
        paths = [(source.parent / p).resolve() for p in element.get('value').split(',')]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing {element.tag} input: {path}")
        inputs.extend(paths)
        if element.tag == 'additional-files':
            copies = []
            for index, path in enumerate(paths):
                tree = ET.parse(path)
                for detector in tree.iter():
                    if 'file' in detector.attrib:
                        if detector.tag not in {'e1Detector', 'e2Detector', 'e3Detector', 'inductionLoop', 'laneAreaDetector', 'entryExitDetector'}:
                            raise ValueError(f"Unrecognised additional file reference: {path}: {detector.tag}")
                        detector.set('file', str(output / Path(detector.get('file')).name))
                copy = output / f'additional_{index}_{path.name}'
                tree.write(copy, encoding='utf-8', xml_declaration=True)
                copies.append(copy)
            paths = copies
        element.set('value', ','.join(map(str, paths)))
    for element in cfg.findall('output/*') + cfg.findall('report/*'):
        if element.tag.endswith('output') or element.tag in {'log', 'message-log', 'error-log'}:
            element.set('value', str(output / Path(element.get('value')).name))
    target = output / 'baseline.sumocfg'
    cfg.write(target, encoding='utf-8', xml_declaration=True)
    return target, inputs


def load_traci(binary):
    candidates = [Path(os.environ.get('SUMO_HOME', '')) / 'tools',
                  Path(binary).resolve().parent.parent / 'tools']
    for path in candidates:
        if (path / 'traci').is_dir():
            sys.path.insert(0, str(path))
            break
    import traci
    return traci


def load_regions(network, region):
    net = ET.parse(network).getroot()
    edges = {e.get('id') for e in net.findall('edge') if e.get('function') != 'internal'}
    region_a = {line.strip().split(':', 1)[1]
                for line in region.read_text(encoding='utf-8-sig').splitlines()
                if line.strip().startswith('edge:')}
    if not region_a or region_a - edges:
        raise ValueError('Region A contains missing or unknown edges')
    return edges, region_a


def measure_accumulation(traci, edges, region_a):
    counts = {'A': 0, 'B': 0}
    for vehicle in traci.vehicle.getIDList():
        road = traci.vehicle.getRoadID(vehicle)
        if not road:
            continue
        # Internal junction lanes belong to the incoming route edge, as before.
        if road in edges:
            edge = road
        else:
            route = traci.vehicle.getRoute(vehicle)
            index = traci.vehicle.getRouteIndex(vehicle)
            edge = route[index] if 0 <= index < len(route) else None
        if edge in edges:
            counts['A' if edge in region_a else 'B'] += 1
    return counts
