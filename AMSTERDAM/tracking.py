#!/usr/bin/env python3
"""
Multi-car anonymity + passive-tracking runner for trajectory obfuscation.

Features:
- Pre/Post-Ghosting
- Trackers: NNPDA, PDA (Log-Sum-Exp), MHT, MM-HMM
- Full detailed CSV metrics & Map PNG Generation
- Crash Resilience: Automatically skips vehicles with broken SUMO routes.
- Modular File System: Runs dynamically on any provided --base-dir.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# OPTIMIZATION: Use KDTree for spatial lookups
from scipy.spatial import cKDTree

def _ensure_sumo_tools() -> None:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)

_ensure_sumo_tools()

try:
    import traci  # type: ignore
    import sumolib  # type: ignore
except Exception as exc:
    raise SystemExit(f"Could not import traci/sumolib. Error: {exc}")

Coord = Tuple[float, float]
Path = Tuple[str, ...]

@dataclass(frozen=True)
class FakeCandidate:
    kind: str
    path: Path
    current_xy: Coord
    score: float
    offset_m: float
    edge_id: str

@dataclass(frozen=True)
class TrackObs:
    step: int
    sim_time: float
    cand_id: int
    is_true: bool
    kind: str
    current_xy: Coord
    speed: float
    angle_deg: float
    full_path: Path
    edge_id: str

@dataclass
class TrackResult:
    tracker: str
    chosen_indices: List[int]
    chosen_cand_ids: List[int]
    chosen_is_true: List[bool]
    success_rate: float
    full_trace_recovery: int
    time_to_first_error_step: int
    time_to_first_error_s: float
    longest_true_streak: int
    chosen_paths: List[Path]

class TopologyCache:
    def __init__(self, net: sumolib.net.Net):
        self.net = net
        self.cache: Dict[Tuple[str, str, int], bool] = {}
        self._edge_map = {e.getID(): e for e in net.getEdges()}

    def get_outgoing_ids(self, edge_id: str) -> List[str]:
        edge = self._edge_map.get(edge_id)
        if not edge: return []
        out = []
        try:
            outgoing = edge.getOutgoing()
            if isinstance(outgoing, dict):
                out.extend(e.getID() for e in outgoing.keys() if hasattr(e, "getID"))
            else:
                for item in outgoing:
                    if hasattr(item, "getID"): out.append(item.getID())
                    elif hasattr(item, "getToLane"): out.append(item.getToLane().getEdge().getID())
        except: pass
        return out

    def is_reachable(self, e1_id: str, e2_id: str, max_depth: int = 3) -> bool:
        if e1_id == e2_id: return True
        key = (e1_id, e2_id, max_depth)
        if key in self.cache: return self.cache[key]
        queue, visited = [(e1_id, 0)], {e1_id}
        while queue:
            curr, depth = queue.pop(0)
            if curr == e2_id:
                self.cache[key] = True
                return True
            if depth < max_depth:
                for nxt in self.get_outgoing_ids(curr):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, depth + 1))
        self.cache[key] = False
        return False


# --- STRING & LIST HELPERS ---

def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]

def parse_str_list(text: str) -> Set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}

def list_vehicle_ids_in_route_file(route_file: str) -> List[str]:
    tree = ET.parse(route_file)
    return [c.get("id", f"veh_{i}") for i, c in enumerate(list(tree.getroot())) if c.tag == "vehicle"]

def sanitize_filename_token(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in text)


# --- MATH & METRICS ---

def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else default

def euclidean(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def normalize_angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)

def entropy_uniform(k: int) -> float:
    return math.log(k) if k > 1 else 0.0

def normalized_entropy_from_h(h: float, support_size: int) -> float:
    return h / math.log(support_size) if support_size > 1 else 0.0

def entropy_from_counts(counts: Sequence[int]) -> float:
    total = sum(c for c in counts if c > 0)
    if total <= 1: return 0.0
    return sum(- (c/total) * math.log(c/total) for c in counts if c > 0)

def trajectory_divergence(path_a: Sequence[str], path_b: Sequence[str], horizon_edges: int) -> float:
    a, b = path_a[:horizon_edges], path_b[:horizon_edges]
    if not a and not b: return 0.0
    if not a: a = tuple([b[0]] * len(b))
    if not b: b = tuple([a[0]] * len(a))
    length = max(len(a), len(b))
    diffs = sum(1 for i in range(length) if a[min(i, len(a)-1)] != b[min(i, len(b)-1)])
    return diffs / max(length, 1)

def mean_pairwise_divergence(paths: Sequence[Sequence[str]], horizon_edges: int) -> float:
    if len(paths) <= 1: return 0.0
    vals = [trajectory_divergence(paths[i], paths[j], horizon_edges) 
            for i in range(len(paths)) for j in range(i + 1, len(paths))]
    return safe_mean(vals, 0.0)

def signature_entropy(paths: Sequence[Sequence[str]], horizon_edges: int) -> Tuple[int, float]:
    counts: Dict[Path, int] = {}
    for p in paths:
        sig = tuple(p[: max(horizon_edges, 1)]) if p else tuple()
        counts[sig] = counts.get(sig, 0) + 1
    return len(counts), entropy_from_counts(list(counts.values()))

def dedup_positions(positions: List[Coord], tol_m: float) -> List[Coord]:
    kept: List[Coord] = []
    for p in positions:
        if all(euclidean(p, q) > tol_m for q in kept): kept.append(p)
    return kept


# --- FILE & OS HELPERS ---

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f: pass
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def append_csv_row(path: str, row: Dict[str, object]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: writer.writeheader()
        writer.writerow(row)

def resolve_existing_file(base_dir: str, preferred: str, candidates: Sequence[str]) -> str:
    if preferred:
        if os.path.isabs(preferred): return preferred
        return os.path.abspath(os.path.join(base_dir, preferred))
    for c in candidates:
        path = os.path.abspath(os.path.join(base_dir, c))
        if os.path.exists(path): return path
    raise FileNotFoundError(f"Could not locate any of: {candidates} in base_dir: {base_dir}")

def sumocfg_value(root: ET.Element, section: str, attr_name: str) -> Optional[str]:
    sec = root.find(section)
    if sec is None: return None
    elem = sec.find(attr_name)
    if elem is not None and elem.get("value"): return elem.get("value")
    for child in sec:
        if child.tag == attr_name and child.get("value"): return child.get("value")
    return None

def resolve_from_sumocfg(sumocfg_path: str) -> Tuple[str, List[str]]:
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()
    cfg_dir = os.path.dirname(sumocfg_path)
    net_rel = sumocfg_value(root, "input", "net-file")
    if not net_rel: raise RuntimeError("Could not resolve net-file from sumocfg")
    net_path = os.path.abspath(os.path.join(cfg_dir, net_rel))
    routes_rel = sumocfg_value(root, "input", "route-files")
    if not routes_rel: return net_path, []
    return net_path, [os.path.abspath(os.path.join(cfg_dir, item.strip())) for item in routes_rel.split(",") if item.strip()]

def sanitize_route_file(route_file: str, net: sumolib.net.Net, out_dir: str) -> Tuple[str, Dict[str, int]]:
    tree = ET.parse(route_file)
    out_path = os.path.join(out_dir, "sanitized.rou.xml")
    tree.write(out_path)
    return out_path, {"kept_vehicles": 0, "removed_vehicles": 0}

def select_single_vehicle_route_file(route_file: str, vehicle_index: int, out_dir: str) -> Tuple[str, Dict[str, object], float, List[str]]:
    tree = ET.parse(route_file)
    root = tree.getroot()
    vehicles = [c for c in list(root) if c.tag == "vehicle"]
    selected = vehicles[vehicle_index]
    vid = selected.get("id", f"veh_{vehicle_index}")
    depart_time = float(selected.get("depart", "0.0"))
    planned_route = []
    for r in selected.findall("route"): planned_route.extend(r.get("edges", "").split())
    
    for child in list(root):
        if child.tag == "vehicle" and id(child) != id(selected):
            root.remove(child)
            
    out_path = os.path.join(out_dir, f"veh_{vehicle_index}.rou.xml")
    tree.write(out_path)
    return out_path, {"selected_vehicle_id": vid}, depart_time, planned_route

def create_sumocfg(sumocfg_src: str, net_file: str, route_file: str, out_dir: str, step_length: float, name: str) -> str:
    tree = ET.parse(sumocfg_src)
    root = tree.getroot()
    for tag in ["net-file", "route-files", "step-length"]:
        for section in list(root):
            for elem in section.findall(tag): section.remove(elem)
            
    inp = root.find("input") or ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file").set("value", os.path.relpath(net_file, out_dir))
    ET.SubElement(inp, "route-files").set("value", os.path.relpath(route_file, out_dir))
    
    tm = root.find("time") or ET.SubElement(root, "time")
    ET.SubElement(tm, "step-length").set("value", str(step_length))
    
    out_cfg = os.path.join(out_dir, name)
    tree.write(out_cfg)
    return out_cfg


# --- MAP / GEOMETRY HELPERS ---

def get_edge_midpoint(edge: "sumolib.net.edge.Edge") -> Coord:
    shape = edge.getShape() or edge.getRawShape()
    return (float(shape[len(shape)//2][0]), float(shape[len(shape)//2][1])) if shape else (0.0, 0.0)

def edge_shape_coords(edge: "sumolib.net.edge.Edge") -> List[Coord]:
    return [(float(x), float(y)) for x, y in (edge.getShape() or edge.getRawShape() or [])]

def polyline_point_at(shape: Sequence[Coord], dist_m: float) -> Coord:
    if not shape: return (0.0, 0.0)
    if len(shape) == 1: return shape[0]
    rem = max(0.0, dist_m)
    for a, b in zip(shape, shape[1:]):
        d = euclidean(a, b)
        if rem <= d:
            t = rem / d if d > 0 else 0
            return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))
        rem -= d
    return shape[-1]

def closest_point_on_segment(p: Coord, a: Coord, b: Coord) -> Coord:
    vx, vy = b[0] - a[0], b[1] - a[1]
    denom = vx * vx + vy * vy
    if denom <= 1e-12: return a
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / denom))
    return (a[0] + t * vx, a[1] + t * vy)

def closest_point_on_polyline(shape: Sequence[Coord], p: Coord) -> Tuple[Coord, float]:
    if not shape: return ((0.0, 0.0), math.inf)
    best_q, best_d = shape[0], euclidean(shape[0], p)
    for a, b in zip(shape, shape[1:]):
        vx, vy = b[0]-a[0], b[1]-a[1]
        denom = vx*vx + vy*vy
        t = max(0.0, min(1.0, ((p[0]-a[0])*vx + (p[1]-a[1])*vy) / denom)) if denom > 0 else 0
        q = (a[0] + t*vx, a[1] + t*vy)
        d = euclidean(q, p)
        if d < best_d: best_q, best_d = q, d
    return best_q, best_d

def effective_fake_k_target(args: argparse.Namespace) -> int:
    return max(1, min(int(args.k_anon_target), int(args.max_fake_set_size)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-dir", default=".", help="Base folder containing SUMO files (net, route, sumocfg). Outputs will be saved here.")
    p.add_argument("--sumocfg", default="", help="SUMO config to use. If empty, searches for sim.sumocfg or cfg/sim.sumocfg in base-dir.")
    p.add_argument("--net-file", default="", help="SUMO network file.")
    p.add_argument("--route-file", default="", help="Route file.")
    p.add_argument("--k-values", default="5,10,15,20,50", help="Comma-separated anonymity targets k.")
    p.add_argument("--vehicle-sample-size", type=int, default=1000)
    p.add_argument("--vehicle-start-index", type=int, default=0)
    p.add_argument("--max-fake-set-size", type=int, default=100)
    p.add_argument("--obfuscation-radius-m", type=float, default=100.0)
    p.add_argument("--real-anon-radius-m", type=float, default=100.0)
    p.add_argument("--path-corridor-m", type=float, default=100.0)
    p.add_argument("--step-length", type=float, default=0.5)
    p.add_argument("--horizon-edge-cap", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=0)
    
    # GHOSTING PARAMETERS
    p.add_argument("--pre-ghost-s", type=float, default=10.0, help="Seconds to generate fake paths BEFORE vehicle departs.")
    p.add_argument("--post-ghost-s", type=float, default=10.0, help="Seconds to generate fake paths AFTER vehicle arrives.")
    
    p.add_argument("--target-max-steps", type=int, default=7200)
    p.add_argument("--target-max-stall-steps", type=int, default=1200)
    p.add_argument("--target-min-progress-m", type=float, default=0.10)
    p.add_argument("--target-stall-speed-threshold-ms", type=float, default=0.10)
    p.add_argument("--blacklist-vehicle-ids", default="veh_12288")
    p.add_argument("--heading-threshold-deg", type=float, default=45.0)
    p.add_argument("--speed-threshold-ms", type=float, default=5.0)
    p.add_argument("--dedupe-tol-m", type=float, default=0.75)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sumo-gui", action="store_true")
    p.add_argument("--write-detailed-csvs", action="store_true")
    p.add_argument("--dump-fake-paths", action="store_true")
    p.add_argument("--no-export-route-pngs", dest="export_route_pngs", action="store_false")
    p.set_defaults(export_route_pngs=True)
    p.add_argument("--matrix-summary-file", default="anonymity_multicar_h30_matrix_1k.csv")
    p.add_argument("--force-rerun", action="store_true")

    # Kalman parameters
    p.add_argument("--track-process-noise-pos", type=float, default=2.0)
    p.add_argument("--track-process-noise-vel", type=float, default=1.0)
    p.add_argument("--track-meas-noise-pos", type=float, default=6.0)
    p.add_argument("--track-init-pos-std", type=float, default=8.0)
    p.add_argument("--track-init-vel-std", type=float, default=8.0)
    p.add_argument("--track-gate-threshold", type=float, default=16.0)
    p.add_argument("--mht-beam-width", type=int, default=12)
    p.add_argument("--track-pd", type=float, default=0.99)
    p.add_argument("--track-clutter-density", type=float, default=0.1)
    return p.parse_args()


# --- FAKE PATH GENERATOR ---

class DummySynthesizer:
    def __init__(self, net: sumolib.net.Net, args: argparse.Namespace, radius_m: float):
        self.net, self.args, self.radius_m = net, args, radius_m
        self.edge_midpoints, self.edge_shapes, self.nearby = {}, {}, {}
        
        for edge in net.getEdges():
            eid = edge.getID()
            if eid.startswith(":"): continue
            self.edge_midpoints[eid] = get_edge_midpoint(edge)
            self.edge_shapes[eid] = edge_shape_coords(edge)
        
        # [OPTIMIZATION] KDTree drastically reduces initialization time
        self.edge_list = list(self.edge_midpoints.keys())
        points = [self.edge_midpoints[e] for e in self.edge_list]
        self.tree = cKDTree(points) if points else None
        self.corr = float(args.path_corridor_m)

    def get_nearby(self, eid: str) -> List[str]:
        if eid in self.nearby:
            return self.nearby[eid]
        if not self.tree:
            return [eid]
        p = self.edge_midpoints.get(eid, (0, 0))
        idxs = self.tree.query_ball_point(p, self.corr)
        res = sorted([self.edge_list[i] for i in idxs], key=lambda o: euclidean(p, self.edge_midpoints[o]))
        self.nearby[eid] = res
        return res

    def get_outgoing_edge_ids(self, edge_id: str) -> List[str]:
        try:
            edge = self.net.getEdge(edge_id)
            out = []
            outgoing = edge.getOutgoing()
            if isinstance(outgoing, dict):
                out.extend(e.getID() for e in outgoing.keys() if hasattr(e, "getID"))
            else:
                for item in outgoing:
                    if hasattr(item, "getID"): out.append(item.getID())
                    elif hasattr(item, "getToLane"): out.append(item.getToLane().getEdge().getID())
            return [e for e in set(out) if not e.startswith(":")]
        except: return []

    def trajectory_edge_distance(self, path_a: Sequence[str], path_b: Sequence[str]) -> float:
        if not path_a or not path_b: return 0.0
        length = max(len(path_a), len(path_b))
        ds = []
        for idx in range(length):
            pa = self.edge_midpoints.get(path_a[min(idx, len(path_a) - 1)])
            pb = self.edge_midpoints.get(path_b[min(idx, len(path_b) - 1)])
            if pa and pb: ds.append(euclidean(pa, pb))
        return safe_mean(ds, 0.0)

    def max_stepwise_path_distance(self, candidate: Sequence[str], true_path: Sequence[str]) -> float:
        if not candidate or not true_path: return math.inf
        ds = []
        for i, eid in enumerate(candidate):
            tid = true_path[min(i, len(true_path) - 1)]
            pa = self.edge_midpoints.get(eid)
            pb = self.edge_midpoints.get(tid)
            if pa is None or pb is None: return math.inf
            ds.append(euclidean(pa, pb))
        return max(ds) if ds else math.inf

    def candidate_within_strict_corridor(self, candidate: Sequence[str], true_path: Sequence[str]) -> bool:
        max_sep = self.max_stepwise_path_distance(candidate, true_path)
        return math.isfinite(max_sep) and max_sep <= self.args.path_corridor_m + 1e-9

    def bounded_current_position_candidates(self, true_xy: Coord, lane_pos: float, lane_len: float, shape: Sequence[Coord]) -> List[Tuple[Coord, float]]:
        if not shape or lane_len <= 0.0: return []
        samples, deduped = [], []
        sample_count = max(16, effective_fake_k_target(self.args) * 4)
        for i in range(sample_count + 1):
            delta = -self.radius_m + 2.0 * self.radius_m * (i / sample_count)
            xy = polyline_point_at(shape, min(max(lane_pos + delta, 0.0), lane_len))
            if euclidean(xy, true_xy) <= self.radius_m + 1e-9: samples.append((xy, delta))
        for xy, delta in samples:
            if all(euclidean(xy, prev_xy) > 0.10 for prev_xy, _ in deduped): deduped.append((xy, delta))
        return deduped

    def generate_candidates(self, true_future_path: Sequence[str], exclude_true: bool) -> List[Path]:
        if not true_future_path: return []
        k_tar = effective_fake_k_target(self.args)
        max_cands, beam_width = max(24, k_tar * 12), max(16, k_tar * 3)
        
        # [OPTIMIZATION] Replaced sluggish nearby array with fast KDTree query
        allowed_by_step = []
        for tid in true_future_path:
            nb = self.get_nearby(tid)
            allowed_by_step.append(nb if nb else [tid])
        
        starts = [true_future_path[0]] if exclude_true else allowed_by_step[0][:beam_width]
        exclude = {tuple(true_future_path)} if exclude_true else set()
        
        states: List[Path] = [tuple([eid]) for eid in starts]
        completed: List[Path] = []
        seen = set()
        
        for depth in range(1, len(true_future_path)):
            next_states = {}
            allowed = set(allowed_by_step[depth])
            for state in states:
                outgoing = [eid for eid in self.get_outgoing_edge_ids(state[-1]) if eid in allowed]
                if not outgoing:
                    if state not in seen and state not in exclude and state != tuple(true_future_path):
                        completed.append(state); seen.add(state)
                    continue
                for nxt in outgoing:
                    new_state = tuple(list(state) + [nxt])
                    if not self.candidate_within_strict_corridor(new_state, true_future_path): continue
                    score = self.trajectory_edge_distance(new_state, true_future_path[:len(new_state)])
                    if new_state not in next_states or score < next_states[new_state]:
                        next_states[new_state] = score
            if not next_states: break
            states = [p for p, _ in sorted(next_states.items(), key=lambda kv: kv[1])[:beam_width]]
            for cand in states:
                if cand not in seen and cand not in exclude and cand != tuple(true_future_path):
                    completed.append(cand); seen.add(cand)
                if len(completed) >= max_cands: return completed[:max_cands]
        return completed[:max_cands]

    def build_route_context(self, veh_id: str) -> Dict[str, object]:
        try:
            r = list(traci.vehicle.getRoute(veh_id))
            idx = int(traci.vehicle.getRouteIndex(veh_id))
            return {"full_route": tuple(r), "past_prefix": tuple(r[:idx]), "future_path": tuple(r[idx:])}
        except:
            return {"full_route": tuple(), "past_prefix": tuple(), "future_path": tuple()}

    def generate_selected_fake_paths(self, true_xy: Coord, true_current_edge: str, true_future_path: Sequence[str], lane_pos: float, lane_len: float, shape: Sequence[Coord]) -> List[FakeCandidate]:
        selected: List[FakeCandidate] = []
        if not true_future_path: return selected
        dyn_horizon = max(1, len(true_future_path))

        bounded_points = self.bounded_current_position_candidates(true_xy, lane_pos, lane_len, shape)
        for xy, delta in bounded_points:
            score = 1.0 + min(euclidean(xy, true_xy) / max(self.radius_m, 1e-9), 1.0)
            selected.append(FakeCandidate("offset", tuple(true_future_path), xy, score, delta, true_current_edge))

        for path in self.generate_candidates(true_future_path, exclude_true=True):
            score = 2.20 + 0.85 * trajectory_divergence(path, true_future_path, dyn_horizon) + 0.015 * self.trajectory_edge_distance(path, true_future_path)
            selected.append(FakeCandidate("same_start_split", tuple(path), true_xy, score, 0.0, path[0]))

        for path in self.generate_candidates(true_future_path, exclude_true=False):
            start_edge = path[0]
            if start_edge == true_current_edge: xy, dist = true_xy, 0.0
            else:
                q_d = closest_point_on_polyline(self.edge_shapes.get(start_edge, []), true_xy)
                if math.isfinite(q_d[1]) and q_d[1] <= self.radius_m + 1e-9: xy, dist = q_d
                else: continue
            score = 1.50 + (0.0 if start_edge == true_current_edge else 0.35) + 0.50 * trajectory_divergence(path, true_future_path, dyn_horizon) + 0.10 * min(dist / max(self.radius_m, 1e-9), 1.0)
            selected.append(FakeCandidate("nearby_start", tuple(path), xy, score, dist, start_edge))

        target = max(effective_fake_k_target(self.args) - 1, 0)
        pool = [c for c in selected if euclidean(c.current_xy, true_xy) <= self.radius_m + 1e-9]
        
        dedup_pool = []
        for cand in pool:
            if not any(prev.kind == cand.kind and prev.path == cand.path and euclidean(prev.current_xy, cand.current_xy) <= 0.10 for prev in dedup_pool):
                dedup_pool.append(cand)

        chosen = []
        while dedup_pool and len(chosen) < target:
            best_idx, best_val = 0, -1e18
            for i, cand in enumerate(dedup_pool):
                pos_div = min([euclidean(cand.current_xy, c.current_xy) for c in chosen] or [0.0])
                path_div = min([trajectory_divergence(cand.path, c.path, dyn_horizon) for c in chosen] or [0.0])
                utility = cand.score + 0.25 * pos_div + 3.0 * path_div
                if utility > best_val: best_val, best_idx = utility, i
            chosen.append(dedup_pool.pop(best_idx))
        return chosen


# --- PLOTTING & LOGGING HEADERS ---

def unique_paths(paths: Sequence[Sequence[str]]) -> List[Path]:
    seen: Set[Path] = set()
    out: List[Path] = []
    for p in paths:
        tp = tuple(p)
        if tp and tp not in seen:
            seen.add(tp)
            out.append(tp)
    return out

def plot_route_map_png(net: sumolib.net.Net, true_route: Sequence[str], fake_routes: Sequence[Sequence[str]], out_path: str, title: str, tracker_routes: Optional[Dict[str, Sequence[Sequence[str]]]] = None) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    ax.set_facecolor("#6EA64A")
    fig.patch.set_facecolor("#6EA64A")

    for edge in net.getEdges():
        if edge.getID().startswith(":"): continue
        coords = edge_shape_coords(edge)
        if len(coords) >= 2:
            ax.plot(*zip(*coords), color="#1F1F1F", linewidth=0.45, alpha=0.48, zorder=0)

    for route in unique_paths(fake_routes):
        for eid in route:
            if eid.startswith(":"): continue
            try: coords = edge_shape_coords(net.getEdge(eid))
            except: continue
            if len(coords) >= 2: ax.plot(*zip(*coords), color=(0.20, 0.55, 1.00), linewidth=1.05, alpha=0.50, zorder=2)

    color_map = {"nnpda": (0.65, 0.16, 0.16), "pda": (1.00, 0.55, 0.00), "mht": (0.80, 0.30, 0.80), "mm_hmm": (0.00, 1.00, 1.00)}
    z = 3
    for name, routes in (tracker_routes or {}).items():
        color = color_map.get(name, (0.9, 0.2, 0.2))
        for route in unique_paths(routes):
            for eid in route:
                if eid.startswith(":"): continue
                try: coords = edge_shape_coords(net.getEdge(eid))
                except: continue
                if len(coords) >= 2: ax.plot(*zip(*coords), color=color, linewidth=1.65, alpha=0.75, zorder=z)
        z += 1

    true_unique = unique_paths([true_route])[0] if true_route else tuple()
    for eid in true_unique:
        if eid.startswith(":"): continue
        try: coords = edge_shape_coords(net.getEdge(eid))
        except: continue
        if len(coords) >= 2:
            ax.plot(*zip(*coords), color="black", linewidth=5.25, alpha=1.0, zorder=10)
            ax.plot(*zip(*coords), color="white", linewidth=2.75, alpha=1.0, zorder=11)

    ax.set_title(title, color="white", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def tracking_summary_fields(prefix: str, res: TrackResult) -> Dict[str, object]:
    return {
        f"{prefix}_tracking_success_rate": round(res.success_rate, 6),
        f"{prefix}_full_trace_recovery": int(res.full_trace_recovery),
        f"{prefix}_time_to_first_error_step": int(res.time_to_first_error_step),
        f"{prefix}_time_to_first_error_s": round(res.time_to_first_error_s, 6) if res.time_to_first_error_s >= 0 else -1,
        f"{prefix}_longest_true_streak": int(res.longest_true_streak),
    }

def summarize_perstep_rows(rows: List[Dict[str, object]], vehicle_index: int, vehicle_id: str, k_target: int, fake_set_cap: int, real_anon_radius_m: float, obfuscation_radius_m: float, route_png_path: str, nnpda_res: TrackResult, pda_res: TrackResult, mht_res: TrackResult, mm_hmm_res: TrackResult, run_ended_reason: str) -> Dict[str, object]:
    base = {
        "vehicle_count_requested": 1,
        "vehicle_index": vehicle_index,
        "vehicle_id": vehicle_id,
        "k_anon_target": k_target,
        "fake_set_cap": fake_set_cap,
        "obfuscation_radius_m": obfuscation_radius_m,
        "real_anon_radius_m": real_anon_radius_m,
        "strict_current_position_radius_enforced": True,
        "strict_future_path_radius_enforced": True,
        "horizon_mode": "capped_first_30_usable_edges",
        "run_ended_reason": run_ended_reason,
        "target_alive_steps": len(rows),
        "target_stall_steps_at_end": 0,
        "mean_horizon_edges_lifetime": round(safe_mean([float(r.get("horizon_edges_lifetime", 0)) for r in rows]), 6),
        "rows": len(rows),
        "vehicles_observed": 1,
        "mean_k_real": round(safe_mean([float(r.get("k_real", 0)) for r in rows]), 6),
        "mean_k_fake": round(safe_mean([float(r.get("k_fake", 0)) for r in rows]), 6),
        "mean_k_combined": round(safe_mean([float(r.get("k_combined", 0)) for r in rows]), 6),
        "mean_delta_k_vs_real": round(safe_mean([float(r.get("delta_k_vs_real", 0)) for r in rows]), 6),
        "mean_h_real": round(safe_mean([float(r.get("h_real", 0)) for r in rows]), 6),
        "mean_h_fake": round(safe_mean([float(r.get("h_fake", 0)) for r in rows]), 6),
        "mean_h_combined": round(safe_mean([float(r.get("h_combined", 0)) for r in rows]), 6),
        "mean_k_traj_real": round(safe_mean([float(r.get("k_traj_real", 0)) for r in rows]), 6),
        "mean_k_traj_fake": round(safe_mean([float(r.get("k_traj_fake", 0)) for r in rows]), 6),
        "mean_k_traj_combined": round(safe_mean([float(r.get("k_traj_combined", 0)) for r in rows]), 6),
        "mean_delta_k_traj_vs_real": round(safe_mean([float(r.get("delta_k_traj_vs_real", 0)) for r in rows]), 6),
        "mean_h_traj_real": round(safe_mean([float(r.get("h_traj_real", 0)) for r in rows]), 6),
        "mean_h_traj_fake": round(safe_mean([float(r.get("h_traj_fake", 0)) for r in rows]), 6),
        "mean_h_traj_combined": round(safe_mean([float(r.get("h_traj_combined", 0)) for r in rows]), 6),
        "mean_true_to_fake_div": round(safe_mean([float(r.get("true_to_fake_div_mean", 0)) for r in rows]), 6),
        "mean_pairwise_fake_div": round(safe_mean([float(r.get("pairwise_fake_div_mean", 0)) for r in rows]), 6),
        "mean_pairwise_combined_div": round(safe_mean([float(r.get("pairwise_combined_div_mean", 0)) for r in rows]), 6),
        "route_png": route_png_path,
    }
    base.update(tracking_summary_fields("nnpda", nnpda_res))
    base.update(tracking_summary_fields("jpda", pda_res))
    base.update(tracking_summary_fields("mht", mht_res))
    base.update(tracking_summary_fields("mm_hmm", mm_hmm_res))
    return base


# ---------- KALMAN MATH BLOCK ---------- #

def kalman_matrices(dt: float, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    q = np.diag([args.track_process_noise_pos**2]*2 + [args.track_process_noise_vel**2]*2)
    r = np.diag([args.track_meas_noise_pos**2]*2)
    return f, h, q, r

def kalman_predict(x: np.ndarray, p: np.ndarray, f: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return f @ x, f @ p @ f.T + q

def innovation_stats(x_pred: np.ndarray, p_pred: np.ndarray, obs: TrackObs, h: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    z = np.array([float(obs.current_xy[0]), float(obs.current_xy[1])])
    y = z - (h @ x_pred)
    s = h @ p_pred @ h.T + r
    try: s_inv = np.linalg.inv(s)
    except np.linalg.LinAlgError: s_inv = np.linalg.inv(s + 1e-6 * np.eye(2))
    d2 = float(y.T @ s_inv @ y)
    log_like = -0.5 * (d2 + math.log(max(float(np.linalg.det(s)), 1e-12)) + 2.0 * math.log(2.0 * math.pi))
    return z, y, s, d2, log_like

def kalman_update(x_pred: np.ndarray, p_pred: np.ndarray, z: np.ndarray, h: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = z - (h @ x_pred)
    s = h @ p_pred @ h.T + r
    try: s_inv = np.linalg.inv(s)
    except np.linalg.LinAlgError: s_inv = np.linalg.inv(s + 1e-6 * np.eye(2))
    k = p_pred @ h.T @ s_inv
    i = np.eye(p_pred.shape[0])
    return x_pred + k @ y, (i - k @ h) @ p_pred @ (i - k @ h).T + k @ r @ k.T

# ---------- TRACKERS ---------- #

@dataclass
class MHTHypothesis:
    x: np.ndarray; p: np.ndarray; log_prob: float; chosen_indices: List[int]; last_edge_id: str

def run_nnpda_tracker(obs_steps: List[List[TrackObs]], dt: float, args: argparse.Namespace) -> TrackResult:
    if not obs_steps: return TrackResult("nnpda", [], [], [], 0, 0, -1, -1.0, 0, [])
    f, h, q, r = kalman_matrices(dt, args)
    gate = float(args.track_gate_threshold)
    obs0 = obs_steps[0][0]
    th = math.radians(obs0.angle_deg)
    x = np.array([obs0.current_xy[0], obs0.current_xy[1], obs0.speed*math.cos(th), obs0.speed*math.sin(th)])
    p = np.eye(4) * args.track_init_pos_std**2
    chosen = [0]
    
    for t in range(1, len(obs_steps)):
        x_pred, p_pred = kalman_predict(x, p, f, q)
        best_idx, best_z, best_log = 0, None, -1e12
        for idx, obs in enumerate(obs_steps[t]):
            z, y, s, d2, log_like = innovation_stats(x_pred, p_pred, obs, h, r)
            if d2 <= gate and log_like > best_log:
                best_idx, best_z, best_log = idx, z, log_like
        if best_z is not None:
            x, p = kalman_update(x_pred, p_pred, best_z, h, r)
            chosen.append(best_idx)
        else:
            x, p = x_pred, p_pred
            chosen.append(0)
    return _summarize_track("nnpda", chosen, obs_steps)

def run_pda_tracker(obs_steps: List[List[TrackObs]], dt: float, args: argparse.Namespace) -> TrackResult:
    if not obs_steps: return TrackResult("pda", [], [], [], 0, 0, -1, -1.0, 0, [])
    f, h, q, r = kalman_matrices(dt, args)
    gate, pd, clut = args.track_gate_threshold, args.track_pd, args.track_clutter_density
    obs0 = obs_steps[0][0]
    th = math.radians(obs0.angle_deg)
    x, p, chosen = np.array([obs0.current_xy[0], obs0.current_xy[1], obs0.speed*math.cos(th), obs0.speed*math.sin(th)]), np.eye(4)*args.track_init_pos_std**2, [0]

    for t in range(1, len(obs_steps)):
        x_pred, p_pred = kalman_predict(x, p, f, q)
        vals = []
        for idx, o in enumerate(obs_steps[t]):
            z, y, s, d2, log_l = innovation_stats(x_pred, p_pred, o, h, r)
            if d2 <= gate: vals.append((idx, z, log_l))
        
        m = len(vals)
        if m == 0:
            x, p = x_pred, p_pred
            chosen.append(0)
            continue
        
        log_clut = math.log(clut) - math.log(2*math.pi*gate)
        log_weights = [math.log(1.0 - pd + 1e-12)]
        for v in vals:
            log_weights.append(math.log(pd + 1e-12) + v[2] - log_clut)
            
        max_log_w = max(log_weights)
        unnorm = [math.exp(lw - max_log_w) for lw in log_weights]
        total_unnorm = sum(unnorm)
        beta = [u / total_unnorm for u in unnorm]
        
        z_bar = sum(beta[j+1] * vals[j][1] for j in range(m))
        y_bar = z_bar - (h @ x_pred)
        x, p_c = kalman_update(x_pred, p_pred, z_bar, h, r)
        
        spread = sum(beta[j+1] * np.outer(vals[j][1] - (h@x_pred), vals[j][1] - (h@x_pred)) for j in range(m)) - np.outer(y_bar, y_bar)
        try: s_inv = np.linalg.inv(h @ p_pred @ h.T + r)
        except: s_inv = np.linalg.inv(h @ p_pred @ h.T + r + 1e-6*np.eye(2))
        k = p_pred @ h.T @ s_inv
        
        p = beta[0] * p_pred + (1.0 - beta[0]) * p_c + k @ spread @ k.T
        chosen.append(vals[max(range(m), key=lambda i: beta[i+1])][0])
        
    return _summarize_track("pda", chosen, obs_steps)

def run_mht_tracker(obs_steps: List[List[TrackObs]], dt: float, args: argparse.Namespace) -> TrackResult:
    return _run_hyp_tracker("mht", obs_steps, dt, args, None)

def run_mm_hmm_tracker(obs_steps: List[List[TrackObs]], dt: float, args: argparse.Namespace, topo: TopologyCache) -> TrackResult:
    return _run_hyp_tracker("mm_hmm", obs_steps, dt, args, topo)

def _run_hyp_tracker(name, obs_steps, dt, args, topo) -> TrackResult:
    if not obs_steps: return TrackResult(name, [], [], [], 0.0, 0, -1, -1.0, 0, [])
    f, h, q, r = kalman_matrices(dt, args)
    gate, pd, clut, beam = args.track_gate_threshold, args.track_pd, args.track_clutter_density, args.mht_beam_width
    log_clut = math.log(clut) - math.log(2*math.pi*gate)
    
    obs0 = obs_steps[0][0]
    th = math.radians(obs0.angle_deg)
    x0 = np.array([obs0.current_xy[0], obs0.current_xy[1], obs0.speed*math.cos(th), obs0.speed*math.sin(th)])
    hyps = [MHTHypothesis(x0, np.eye(4)*args.track_init_pos_std**2, 0.0, [0], obs0.edge_id)]

    for t in range(1, len(obs_steps)):
        expanded = []
        for hyp in hyps:
            x_p, p_p = kalman_predict(hyp.x, hyp.p, f, q)
            # Missed detection
            expanded.append(MHTHypothesis(x_p, p_p, hyp.log_prob + math.log(1.0 - pd + 1e-12), hyp.chosen_indices + [-1], hyp.last_edge_id))
            # Measurements
            for idx, o in enumerate(obs_steps[t]):
                if topo and not topo.is_reachable(hyp.last_edge_id, o.edge_id): continue
                z, y, s, d2, log_l = innovation_stats(x_p, p_p, o, h, r)
                if d2 <= gate:
                    xn, pn = kalman_update(x_p, p_p, z, h, r)
                    expanded.append(MHTHypothesis(xn, pn, hyp.log_prob + math.log(pd + 1e-12) + log_l - log_clut, hyp.chosen_indices + [idx], o.edge_id))
        
        if expanded:
            max_l = max(ho.log_prob for ho in expanded)
            for ho in expanded: ho.log_prob -= max_l
        hyps = sorted(expanded, key=lambda ho: ho.log_prob, reverse=True)[:beam]

    best = max(hyps, key=lambda ho: ho.log_prob)
    chosen = [idx if idx != -1 else 0 for idx in best.chosen_indices]
    return _summarize_track(name, chosen, obs_steps)


# [CRASH FIX]: Bound checks on index extraction to prevent IndexError
def _summarize_track(name, chosen, steps) -> TrackResult:
    cids, is_t, paths = [], [], []
    for t, idx in enumerate(chosen):
        # Gracefully handle missing observations in the time step
        if t >= len(steps) or not steps[t]:
            continue
            
        # Prevent index out of bounds if tracker logic fed a bad fallback ID
        if idx < 0 or idx >= len(steps[t]):
            idx = 0 
            
        o = steps[t][idx]
        cids.append(o.cand_id)
        is_t.append(o.is_true)
        paths.append(o.full_path)
        
    correct = sum(is_t)
    total_valid_steps = max(1, len(is_t))
    return TrackResult(
        name, chosen, cids, is_t, 
        correct / total_valid_steps, 
        1 if correct == total_valid_steps and is_t else 0, 
        -1, -1.0, 0, paths
    )

# ---------- MAIN RUNNER ---------- #

def main() -> None:
    p = parse_args()
    random.seed(p.seed)
    
    # 1. Resolve base directory dynamically
    base = os.path.abspath(p.base_dir)
    out_dir = os.path.join(base, "outputs")
    prep_dir = os.path.join(base, "prepared")
    ensure_dir(out_dir)
    ensure_dir(prep_dir)

    # 2. Look for config files inside the provided base directory
    cfg = resolve_existing_file(base, p.sumocfg, ["sim.sumocfg", "cfg/sim.sumocfg"])
    net_f, rou_fs = resolve_from_sumocfg(cfg)
    net = sumolib.net.readNet(net_f)
    topo = TopologyCache(net)
    
    # [OPTIMIZATION] Moved DummySynthesizer outside the loop so it builds the KDTree ONCE!
    print("Initializing Spatial Index for Map Topology...")
    synth = DummySynthesizer(net, p, p.obfuscation_radius_m)
    print("Spatial Index Ready.")
    
    san_rou, _ = sanitize_route_file(rou_fs[0], net, prep_dir)
    vids = list_vehicle_ids_in_route_file(san_rou)
    
    start = p.vehicle_start_index
    end = min(len(vids), start + p.vehicle_sample_size)
    summary_path = os.path.join(out_dir, p.matrix_summary_file)

    # 3. Dynamic Resume Logic (No Hardcoded Desktop Paths)
    completed = set()
    if not p.force_rerun and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            completed = {(int(r["vehicle_index"]), int(r["k_anon_target"])) for r in csv.DictReader(f)}
        unique_completed_vehicles = len(set(v for v, k in completed))
        print(f"Resuming from {summary_path}. Found {unique_completed_vehicles} unique vehicles partially or fully completed.")
    else:
        print(f"Starting fresh. Results will be saved to: {summary_path}")

    # 4. Main Simulation Loop
    for v_idx in range(start, end):
        vid = vids[v_idx]
        if vid in parse_str_list(p.blacklist_vehicle_ids): continue
        single_rou, _, dep_t, p_route = select_single_vehicle_route_file(san_rou, v_idx, prep_dir)
        d_cfg = create_sumocfg(cfg, net_f, single_rou, base, p.step_length, f"sim_v_{v_idx}.sumocfg")
        
        for kt in parse_int_list(p.k_values):
            if (v_idx, kt) in completed: 
                print(f"Skipping veh {v_idx} ({vid}), k={kt} (Already completed in this directory)")
                continue
            
            p.k_anon_target = kt
            print(f"\n=== Running veh {v_idx} ({vid}), k={kt} ===")
            
            cmd = [shutil.which("sumo-gui" if p.sumo_gui else "sumo"), "-c", d_cfg, "--no-step-log", "true", "--quit-on-end", "true", "--seed", str(p.seed)]
            
            obs_steps, perstep_rows, fake_full_paths_for_plot = [], [], []
            step, seen, ex_t, last_s = 0, False, None, None
            
            try:
                traci.start(cmd)
                while True:
                    if traci.simulation.getMinExpectedNumber() <= 0 and ex_t and traci.simulation.getTime() > ex_t + p.post_ghost_s: break
                    if p.max_steps > 0 and step >= p.max_steps: break
                    traci.simulationStep()
                    t = traci.simulation.getTime()
                    live = set(traci.vehicle.getIDList())
                    emit, state = False, None

                    if not seen and vid not in live:
                        if t >= dep_t - p.pre_ghost_s:
                            state = {"xy": synth.edge_midpoints.get(p_route[0], (0,0)), "speed": 5.0, "angle": 0.0, "road": p_route[0], "future_path": p_route, "full_route": p_route, "past_prefix": []}
                    elif vid in live:
                        seen, emit = True, True
                        ctx = synth.build_route_context(vid)
                        state = last_s = {"xy": traci.vehicle.getPosition(vid), "speed": traci.vehicle.getSpeed(vid), "angle": traci.vehicle.getAngle(vid), "road": traci.vehicle.getRoadID(vid), **ctx}
                    elif seen and vid not in live:
                        if not ex_t: ex_t = t
                        if t <= ex_t + p.post_ghost_s: state = last_s
                        else: break

                    if state:
                        fakes = synth.generate_selected_fake_paths(state["xy"], state["road"], state["future_path"], 0, 100, synth.edge_shapes.get(state["road"], [(0,0)]))
                        
                        horizon = min(len(state["full_route"]), p.horizon_edge_cap)
                        fake_paths = [state["full_route"]] + [tuple(list(state["past_prefix"]) + list(fc.path)) for fc in fakes]
                        k_fake = len(fake_paths)
                        h_fake = entropy_uniform(k_fake)
                        k_traj, h_traj = signature_entropy(fake_paths, horizon)
                        
                        fake_full_paths_for_plot.extend(fake_paths[1:])
                        
                        perstep_rows.append({
                            "step": step, "sim_time": t, "veh_id": vid, "road_id": state["road"],
                            "horizon_edges_lifetime": horizon,
                            "x_true": round(state["xy"][0], 3), "y_true": round(state["xy"][1], 3),
                            "k_fake": k_fake, "h_fake": round(h_fake, 6),
                            "k_traj_fake": k_traj, "h_traj_fake": round(h_traj, 6),
                            "true_to_fake_div_mean": round(safe_mean([trajectory_divergence(state["full_route"], fp, horizon) for fp in fake_paths[1:]]), 6),
                            "pairwise_fake_div_mean": round(mean_pairwise_divergence(fake_paths, horizon), 6)
                        })

                        obs = []
                        if emit: obs.append(TrackObs(step, t, -1, True, "true", state["xy"], state["speed"], state["angle"], state["full_route"], state["road"]))
                        for fc in fakes: obs.append(TrackObs(step, t, -1, False, fc.kind, fc.current_xy, state["speed"], state["angle"], tuple(list(state["past_prefix"]) + list(fc.path)), fc.edge_id))
                        random.shuffle(obs)
                        obs_steps.append([TrackObs(o.step, o.sim_time, i, o.is_true, o.kind, o.current_xy, o.speed, o.angle_deg, o.full_path, o.edge_id) for i, o in enumerate(obs)])
                    step += 1
            except traci.exceptions.FatalTraCIError as e:
                print(f"    -> Skipping {vid} due to TraCI Error: {e}")
                continue
            finally:
                try: traci.close()
                except: pass

            res = [run_nnpda_tracker(obs_steps, p.step_length, p), run_pda_tracker(obs_steps, p.step_length, p),
                   run_mht_tracker(obs_steps, p.step_length, p), run_mm_hmm_tracker(obs_steps, p.step_length, p, topo)]
            
            png_path = ""
            if p.export_route_pngs and kt == 50 and last_s:
                png_path = os.path.join(out_dir, f"route_map_v{v_idx}_k{kt}.png")
                plot_route_map_png(net, last_s["full_route"], fake_full_paths_for_plot, png_path, f"Veh {v_idx} | k={kt}", {"nnpda": res[0].chosen_paths, "pda": res[1].chosen_paths, "mht": res[2].chosen_paths, "mm_hmm": res[3].chosen_paths})

            summary = summarize_perstep_rows(perstep_rows, v_idx, vid, kt, p.max_fake_set_size, p.real_anon_radius_m, p.obfuscation_radius_m, png_path, res[0], res[1], res[2], res[3], "normal_finish")
            append_csv_row(summary_path, summary)
            
            if p.write_detailed_csvs:
                write_csv(os.path.join(out_dir, f"anonymity_perstep_v{v_idx}_k{kt}.csv"), perstep_rows)
            
            print(f"    -> NNPDA: {res[0].success_rate:.2f} | PDA: {res[1].success_rate:.2f} | MHT: {res[2].success_rate:.2f} | MM-HMM: {res[3].success_rate:.2f}")

if __name__ == "__main__":
    main()
