# smart + romper paredes
# Generado parcialmente con ChatGPT

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import heapq


# LOGGER
class SimLogger:
    def __init__(self):
        self.steps = []  # steps
        # snapshots (después de ejecutar todos los steps por turno)
        self.snapshots = []

    def spawn_agent(self, aid, r, c, t):
        self.steps.append({"t": int(t), "type": "spawn_agent",
                           "id": str(aid), "r": int(r), "c": int(c)})

    def move(self, aid, from_pos, to_pos, t):
        self.steps.append({
            "t": int(t), "type": "move", "id": str(aid),
            "from": {"r": int(from_pos[1] + 1), "c": int(from_pos[0] + 1)},
            "to":   {"r": int(to_pos[1] + 1),   "c": int(to_pos[0] + 1)}
        })

    def reveal_poi(self, r, c, kind, t):
        self.steps.append({"t": int(t), "type": "reveal_poi",
                           "r": int(r), "c": int(c), "kind": str(kind)})

    def rescue(self, r, c, t):
        self.steps.append(
            {"t": int(t), "type": "rescue", "r": int(r), "c": int(c)})

    def riot_spread(self, from_pos, to_pos, t):
        self.steps.append({
            "t": int(t), "type": "riot_spread",
            "from": {"r": int(from_pos[1] + 1), "c": int(from_pos[0] + 1)},
            "to":   {"r": int(to_pos[1] + 1),   "c": int(to_pos[0] + 1)}
        })

    def riot_contained(self, r, c, t):
        self.steps.append(
            {"t": int(t), "type": "riot_contained", "r": int(r), "c": int(c)})

    def damage_inc(self, amount, t):
        self.steps.append(
            {"t": int(t), "type": "damage_inc", "amount": int(amount)})

    def break_wall(self, pos1, pos2, t):
        self.steps.append({
            "t": int(t), "type": "break_wall",
            "r1": int(pos1[1] + 1), "c1": int(pos1[0] + 1),
            "r2": int(pos2[1] + 1), "c2": int(pos2[0] + 1)
        })

    # snapshots (estado final por tick)
    def snapshot_tick(self, model, t, include_pois=False, include_riots=True, include_doors=False):
        snap = {"t": int(t)}

        agents = []
        for a in model.schedule.agents:
            if getattr(a, "pos", None) is not None:
                agents.append(
                    {"id": str(a.unique_id), "r": a.pos[1] + 1, "c": a.pos[0] + 1})
        agents.sort(key=lambda x: int(x["id"]))
        snap["agents"] = agents

        if include_riots:
            riots = []
            for (x, y), contents in model.cell_contents.items():
                d = next((c for c in contents if isinstance(c, Disturbance)), None)
                if d:
                    riots.append(
                        {"r": y + 1, "c": x + 1, "severity": d.severity})
            riots.sort(key=lambda p: (p["r"], p["c"]))
            snap["riots"] = riots

        if include_doors:
            doors = []
            for (x, y), contents in model.cell_contents.items():
                for g in contents:
                    if isinstance(g, Gate):
                        doors.append(
                            {"r": y + 1, "c": x + 1, "open": bool(g.is_open)})
            doors.sort(key=lambda d: (d["r"], d["c"]))
            snap["doors"] = doors

        self.snapshots.append(snap)

    def to_simlog(self, result, rescued, lost, damage, meta=None):
        out = {
            "result": result,
            "rescued": int(rescued),
            "lost": int(lost),
            "damage": int(damage),
            "steps": self.steps,
            "snapshots": self.snapshots
        }
        if meta:
            out["meta"] = meta
        return out


# ENTIDADES
class Hostage:
    def __init__(self, unique_id):
        self.unique_id = unique_id


class FalseAlarm:
    def __init__(self, unique_id):
        self.unique_id = unique_id


class Gate:
    def __init__(self, unique_id, is_open=False):
        self.unique_id = unique_id
        self.is_open = is_open


class Disturbance:
    def __init__(self, unique_id, severity='mild'):
        self.unique_id = unique_id
        self.severity = severity
        self.turns_in_current_state = 0


# render matrices
def get_grid_board(model):
    # Matriz H*W por celda
    # 0 vacío, 2 agente, 3 rehén, 4/5/8 disturbio mild/active/grave,
    # 7 falsa alarma, 6 entrada, 9 gate abierta, 10 gate cerrada
    H, W = model.grid.height, model.grid.width
    M = np.zeros((H, W), dtype=np.int32)

    # entradas si están libres
    for (ex, ey) in getattr(model, "entry_points", []):
        if M[ey, ex] == 0:
            M[ey, ex] = 6

    for (x, y), contents in model.cell_contents.items():
        if any(isinstance(c, Hostage) for c in contents):
            M[y, x] = 3
            continue
        d = next((c for c in contents if isinstance(c, Disturbance)), None)
        if d:
            M[y, x] = 8 if d.severity == "grave" else (
                5 if d.severity == "active" else 4)
            continue
        if any(isinstance(c, FalseAlarm) for c in contents):
            M[y, x] = 7
            continue
        g = next((c for c in contents if isinstance(c, Gate)), None)
        if g:
            M[y, x] = 9 if g.is_open else 10
            continue

    # Agentes encima
    for a in model.schedule.agents:
        if getattr(a, "pos", None) is None:
            continue
        x, y = a.pos
        M[y, x] = 2

    return M


# AGENTE INTELIGENTE
# Dijkstra y jerarquía
class TacticalAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        self.action_points = 4
        self.carrying_hostage = False
        self.current_path = []
        self.current_goal = None

    def step(self):
        self.action_points = 4
        while self.action_points > 0:
            if self._evacuate_if_carrying():
                continue
            if self._rescue_hostage():
                continue
            if self._contain_disturbance():
                continue
            if self._investigate_alarm():
                continue
            if self._explore_map():
                continue
            break

    def _evacuate_if_carrying(self):
        if not self.carrying_hostage:
            return False

        if self.pos in self.model.entry_points and self.action_points >= 1:
            self.carrying_hostage = False
            self.model.hostages_rescued += 1
            self.action_points -= 1
            self.current_path, self.current_goal = [], None
            return True

        if not self.current_path or self.current_goal not in self.model.entry_points:
            nearest = self._nearest_entry()
            if nearest:
                self.current_path = self.model.dijkstra_path(self.pos, nearest)
                self.current_goal = nearest
        return self._advance_along_path()

    def _rescue_hostage(self):
        if self.carrying_hostage:
            return False

        here = self.model.get_contents_at(self.pos)
        h = next((c for c in here if isinstance(c, Hostage)), None)
        if h and self.action_points >= 2:
            self.model.reveal_if_needed(self.pos)
            if hasattr(self.model, "logger"):
                self.model.logger.rescue(
                    self.pos[1] + 1, self.pos[0] + 1, t=self.model.turn_counter + 1)
            self.carrying_hostage = True
            self.model.remove_entity(h, self.pos)
            self.action_points -= 2
            self.current_path, self.current_goal = [], None
            return True

        # rehén más cercano
        if not self.current_path or not self._goal_is_hostage():
            target = self._nearest_hostage()
            if target:
                path = self.model.dijkstra_path(self.pos, target)
                if path and len(path) > 1:
                    self.current_path, self.current_goal = path, target
                    return self._advance_along_path()
        else:
            return self._advance_along_path()
        return False

    def _contain_disturbance(self):
        here = self.model.get_contents_at(self.pos)
        d = next((c for c in here if isinstance(c, Disturbance)), None)
        if d:
            cost = 1 if d.severity == "mild" else (
                2 if d.severity == "active" else 999)
            if cost < 999 and self.action_points >= cost:
                self.model.remove_entity(d, self.pos)
                if hasattr(self.model, "logger"):
                    self.model.logger.riot_contained(
                        self.pos[1] + 1, self.pos[0] + 1, t=self.model.turn_counter + 1)
                self.action_points -= cost
                self.current_path, self.current_goal = [], None
                # caché se limpia si el ambiente cambia
                self.model.clear_pathfinding_cache()
                return True

        if not self.current_path or not self._goal_is_containable_disturbance():
            target = self._nearest_containable_disturbance()
            if target:
                path = self.model.dijkstra_path(self.pos, target)
                if path and len(path) > 1:
                    self.current_path, self.current_goal = path, target
                    return self._advance_along_path()
        else:
            return self._advance_along_path()
        return False

    def _investigate_alarm(self):
        here = self.model.get_contents_at(self.pos)
        fa = next((c for c in here if isinstance(c, FalseAlarm)), None)
        if fa and self.action_points >= 1:
            self.model.reveal_if_needed(self.pos)
            self.model.false_alarms_investigated += 1
            self.model.remove_entity(fa, self.pos)
            self.action_points -= 1
            self.current_path, self.current_goal = [], None
            return True

        if not self.current_path or not self._goal_is_alarm():
            target = self._nearest_alarm()
            if target:
                path = self.model.dijkstra_path(self.pos, target)
                if path and len(path) > 1:
                    self.current_path, self.current_goal = path, target
                    return self._advance_along_path()
        else:
            return self._advance_along_path()
        return False

    def _explore_map(self):
        if not self.current_path or not self._goal_is_exploration():
            target = self._nearest_explorable()
            if target:
                path = self.model.dijkstra_path(self.pos, target)
                if path and len(path) > 1:
                    self.current_path, self.current_goal = path, target
                    return self._advance_along_path()
        else:
            return self._advance_along_path()
        return False

    def _advance_along_path(self):
        if not self.current_path or len(self.current_path) <= 1:
            return False
        next_pos = self.current_path[1]

        if not self.model.can_move_to(self.pos, next_pos):
            if self.model.has_wall_between(self.pos, next_pos) and self.action_points >= 2:
                self.model.break_wall_between(self.pos, next_pos)
                self.model.structural_damage += 1
                if hasattr(self.model, "logger"):
                    self.model.logger.break_wall(
                        self.pos, next_pos, t=self.model.turn_counter + 1)
                self.action_points -= 2
                return True
            self.current_path = []
            return False

        cost = 1
        for content in self.model.get_contents_at(next_pos):
            if isinstance(content, Disturbance):
                cost = 2
                break

        if self.action_points >= cost:
            from_pos = self.pos
            self.model.grid.move_agent(self, next_pos)
            if hasattr(self.model, "logger"):
                self.model.logger.move(
                    self.unique_id, from_pos, next_pos, t=self.model.turn_counter + 1)

            self.current_path = self.current_path[1:]
            self.action_points -= cost

            if next_pos == self.current_goal:
                self.current_path, self.current_goal = [], None

            revealed = self.model.reveal_if_needed(self.pos)
            if revealed:
                self.action_points = 0
            return True

        return False

    def _nearest_entry(self):
        best_d, best = float("inf"), None
        for entry in self.model.entry_points:
            d = self.model.dijkstra_distance(self.pos, entry)
            if d < best_d:
                best_d, best = d, entry
        return best

    def _nearest_hostage(self):
        best_d, best = float("inf"), None
        for pos, contents in self.model.cell_contents.items():
            if any(isinstance(c, Hostage) for c in contents):
                d = self.model.dijkstra_distance(self.pos, pos)
                if d < best_d:
                    best_d, best = d, pos
        return best

    def _nearest_containable_disturbance(self):
        best_d, best = float("inf"), None
        for pos, contents in self.model.cell_contents.items():
            for it in contents:
                if isinstance(it, Disturbance) and it.severity in ("mild", "active"):
                    d = self.model.dijkstra_distance(self.pos, pos)
                    if d < best_d:
                        best_d, best = d, pos
        return best

    def _nearest_alarm(self):
        best_d, best = float("inf"), None
        for pos, contents in self.model.cell_contents.items():
            if any(isinstance(c, FalseAlarm) for c in contents):
                d = self.model.dijkstra_distance(self.pos, pos)
                if d < best_d:
                    best_d, best = d, pos
        return best

    def _nearest_explorable(self):
        best_d, best = float("inf"), None
        for x in range(self.model.grid.width):
            for y in range(self.model.grid.height):
                pos = (x, y)
                cont = self.model.get_contents_at(pos)
                if any(isinstance(g, Gate) and not g.is_open for g in cont):
                    continue
                if pos in self.model.revealed_pois:
                    continue
                d = self.model.dijkstra_distance(self.pos, pos)
                if d < best_d:
                    best_d, best = d, pos
        return best

    def _goal_is_hostage(self):
        return self.current_goal is not None and any(
            isinstance(c, Hostage) for c in self.model.get_contents_at(self.current_goal)
        )

    def _goal_is_containable_disturbance(self):
        if self.current_goal is None:
            return False
        for c in self.model.get_contents_at(self.current_goal):
            if isinstance(c, Disturbance) and c.severity in ("mild", "active"):
                return True
        return False

    def _goal_is_alarm(self):
        return self.current_goal is not None and any(
            isinstance(c, FalseAlarm) for c in self.model.get_contents_at(self.current_goal)
        )

    def _goal_is_exploration(self):
        return self.current_goal is not None


# MODELO
class RescueModel(Model):
    def __init__(self, config_path="config.json"):
        super().__init__()
        cfg = self._load_config(config_path)

        rows, cols = cfg["rows"], cfg["cols"]
        # mesa: W=c, H=r
        self.grid = MultiGrid(cols, rows, torus=False)
        self.schedule = RandomActivation(self)
        self.cell_contents = defaultdict(list)
        self.running = True
        self.hostages_rescued = 0
        self.hostages_lost = 0
        self.structural_damage = 0
        self.false_alarms_investigated = 0
        self.turn_counter = 0
        self.next_entity_id = 0
        self.min_hidden_markers = 3
        self.walls = {}
        self.entry_points = []
        self._build_from_config(cfg)
        self.logger = SimLogger()
        self.revealed_pois = set()
        self._distance_cache = {}

        for _ in range(6):
            a = TacticalAgent(self.get_next_id(), self)
            self.schedule.add(a)
            ep = self.random.choice(self.entry_points)
            self.grid.place_agent(a, ep)

        for a in self.schedule.agents:
            if getattr(a, "pos", None) is not None:
                self.logger.spawn_agent(
                    a.unique_id, a.pos[1] + 1, a.pos[0] + 1, t=0)

        self.datacollector = DataCollector(
            model_reporters={"Grid": lambda m: np.array(get_grid_board(m))}
        )

        self.logger.snapshot_tick(
            self, t=0, include_pois=False, include_riots=True, include_doors=False)

    def _load_config(self, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_cells(self, cfg):
        if "cells" in cfg and cfg["cells"]:
            return cfg["cells"]
        rows, cols = cfg["rows"], cfg["cols"]
        out = []
        for line in cfg["cellRows"]:
            parts = line.strip().split()
            if len(parts) != cols:
                raise ValueError(
                    f"línea con {len(parts)} columnas; se esperaban {cols}.")
            out.append(parts)
        if len(out) != rows:
            raise ValueError(
                f"se recibieron {len(out)} filas; se esperaban {rows}.")
        return out

    def _build_from_config(self, cfg):
        cells = self._parse_cells(cfg)
        rows, cols = cfg["rows"], cfg["cols"]

        for r in range(rows):
            for c in range(cols):
                code = cells[r][c]
                if len(code) != 4:
                    raise ValueError(
                        f"celda ({r},{c}) código inválido: {code}")
                x, y = c, r
                self.walls[(x, y)] = {
                    "top":    code[0] == "1",
                    "left":   code[1] == "1",
                    "bottom": code[2] == "1",
                    "right":  code[3] == "1",
                }

        self.entry_points = []
        for e in cfg.get("entries", []):
            r, c = e["r"], e["c"]
            self.entry_points.append((c - 1, r - 1))

        for p in cfg.get("pois", []):
            r, c, kind = p["r"], p["c"], p["kind"]
            pos = (c - 1, r - 1)
            if kind == "v":
                self.cell_contents[pos].append(Hostage(self.get_next_id()))
            else:
                self.cell_contents[pos].append(FalseAlarm(self.get_next_id()))

        for rr in cfg.get("riots", []):
            r, c = rr["r"], rr["c"]
            pos = (c - 1, r - 1)
            self.cell_contents[pos].append(
                Disturbance(self.get_next_id(), "mild"))

        for d in cfg.get("doors", []):
            r1, c1, r2, c2 = d["r1"], d["c1"], d["r2"], d["c2"]
            is_open = bool(d.get("open", False))
            pos1 = (c1 - 1, r1 - 1)
            self.cell_contents[pos1].append(Gate(self.get_next_id(), is_open))

    def get_next_id(self):
        self.next_entity_id += 1
        return self.next_entity_id

    def get_contents_at(self, pos):
        return self.grid.get_cell_list_contents([pos]) + self.cell_contents.get(pos, [])

    def remove_entity(self, entity, pos):
        if pos in self.cell_contents and entity in self.cell_contents[pos]:
            self.cell_contents[pos].remove(entity)
            self.clear_pathfinding_cache()

    def can_move_to(self, from_pos, to_pos):
        x2, y2 = to_pos
        if not (0 <= x2 < self.grid.width and 0 <= y2 < self.grid.height):
            return False

        x1, y1 = from_pos
        dx, dy = x2 - x1, y2 - y1

        w = self.walls.get(from_pos, {})
        if dx == 1 and w.get('right'):
            return False
        elif dx == -1 and w.get('left'):
            return False
        elif dy == 1 and w.get('bottom'):
            return False
        elif dy == -1 and w.get('top'):
            return False

        cont = self.get_contents_at(to_pos)
        if any(isinstance(g, Gate) and not g.is_open for g in cont):
            return False

        return True

    def has_wall_between(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        dx, dy = x2 - x1, y2 - y1
        w = self.walls.get(pos1, {})
        if dx == 1 and w.get('right'):
            return True
        elif dx == -1 and w.get('left'):
            return True
        elif dy == 1 and w.get('bottom'):
            return True
        elif dy == -1 and w.get('top'):
            return True
        return False

    def break_wall_between(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        dx, dy = x2 - x1, y2 - y1

        w1 = self.walls.get(pos1, {}).copy()
        w2 = self.walls.get(pos2, {}).copy()
        changed = False

        if dx == 1 and w1.get('right'):
            w1['right'] = False
            w2['left'] = False
            changed = True
        elif dx == -1 and w1.get('left'):
            w1['left'] = False
            w2['right'] = False
            changed = True
        elif dy == 1 and w1.get('bottom'):
            w1['bottom'] = False
            w2['top'] = False
            changed = True
        elif dy == -1 and w1.get('top'):
            w1['top'] = False
            w2['bottom'] = False
            changed = True

        if changed:
            self.walls[pos1] = w1
            self.walls[pos2] = w2
            self.clear_pathfinding_cache()

    def clear_pathfinding_cache(self):
        self._distance_cache.clear()

    # dijsktra
    def dijkstra_path(self, start, goal):
        if start == goal:
            return [start]

        pq = [(0, start, [start])]
        visited = set()

        while pq:
            dist, pos, path = heapq.heappop(pq)
            if pos in visited:
                continue
            visited.add(pos)
            if pos == goal:
                return path

            for nb in self.grid.get_neighborhood(pos, moore=False, include_center=False):
                if nb in visited:
                    continue
                if not self.can_move_to(pos, nb):
                    continue

                cost = 1
                for it in self.get_contents_at(nb):
                    if isinstance(it, Disturbance):
                        cost = 2
                        break

                heapq.heappush(pq, (dist + cost, nb, path + [nb]))

        return []

    def dijkstra_distance(self, start, goal):
        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]
        if start == goal:
            self._distance_cache[key] = 0
            return 0

        pq = [(0, start)]
        dist = {start: 0}
        visited = set()

        while pq:
            d, pos = heapq.heappop(pq)
            if pos in visited:
                continue
            visited.add(pos)
            if pos == goal:
                self._distance_cache[key] = d
                return d

            for nb in self.grid.get_neighborhood(pos, moore=False, include_center=False):
                if nb in visited:
                    continue
                if not self.can_move_to(pos, nb):
                    continue

                cost = 1
                for it in self.get_contents_at(nb):
                    if isinstance(it, Disturbance):
                        cost = 2
                        break

                nd = d + cost
                if nb not in dist or nd < dist[nb]:
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))

        self._distance_cache[key] = float("inf")
        return float("inf")

    def get_available_cell(self):
        for _ in range(200):
            pos = (self.random.randrange(self.grid.width),
                   self.random.randrange(self.grid.height))
            cont = self.get_contents_at(pos)
            if any(isinstance(c, Gate) and not c.is_open for c in cont):
                continue
            if any(isinstance(c, TacticalAgent) for c in cont):
                continue
            return pos
        return (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height))

    def place_passive_entity(self, entity_class):
        entity = entity_class(self.get_next_id()) if entity_class != Disturbance \
            else Disturbance(self.get_next_id(), "mild")
        pos = self.get_available_cell()
        self.cell_contents[pos].append(entity)
        if isinstance(entity, Disturbance) and hasattr(self, "logger"):
            self.logger.riot_spread(pos, pos, t=self.turn_counter + 1)
        self.clear_pathfinding_cache()

    def count_hidden_markers(self):
        return sum(1 for cont in self.cell_contents.values() for it in cont
                   if isinstance(it, (Hostage, FalseAlarm)))

    def maintain_minimum_markers(self):
        while self.count_hidden_markers() < self.min_hidden_markers:
            cls = self.random.choices(
                [Hostage, FalseAlarm], weights=[0.7, 0.3], k=1)[0]
            self.place_passive_entity(cls)

    def advance_disturbances(self):
        for pos, cont in list(self.cell_contents.items()):
            ds = [d for d in cont if isinstance(d, Disturbance)]
            for d in ds:
                d.turns_in_current_state += 1
                if d.severity == "mild" and d.turns_in_current_state >= 4:
                    d.severity = "active"
                    d.turns_in_current_state = 0
                    self.clear_pathfinding_cache()
                elif d.severity == "active" and d.turns_in_current_state >= 6:
                    d.severity = "grave"
                    self.handle_explosion(pos, cont)

        if self.random.random() < 0.05:
            pos = self.get_available_cell()
            cont = self.get_contents_at(pos)
            if not any(isinstance(c, Disturbance) for c in cont):
                self.cell_contents[pos].append(
                    Disturbance(self.get_next_id(), "mild"))
                if hasattr(self, "logger"):
                    self.logger.riot_spread(pos, pos, t=self.turn_counter + 1)
                self.clear_pathfinding_cache()

    def handle_explosion(self, pos, contents):
        self.structural_damage += 1
        if hasattr(self, "logger"):
            self.logger.damage_inc(1, t=self.turn_counter + 1)

        neighbors = [(pos[0] + dx, pos[1] + dy)
                     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        for nb in neighbors:
            if self.has_wall_between(pos, nb) and self.random.random() < 0.3:
                self.break_wall_between(pos, nb)

        for nb in neighbors:
            if nb in self.cell_contents:
                gates = [g for g in self.cell_contents[nb]
                         if isinstance(g, Gate)]
                for g in gates:
                    if self.random.random() < 0.5:
                        self.remove_entity(g, nb)

        for h in [h for h in contents if isinstance(h, Hostage)]:
            self.hostages_lost += 1
            self.remove_entity(h, pos)

        removed_any = False
        for d in [d for d in contents if isinstance(d, Disturbance)]:
            self.remove_entity(d, pos)
            removed_any = True
        if removed_any and hasattr(self, "logger"):
            self.logger.riot_contained(
                pos[1] + 1, pos[0] + 1, t=self.turn_counter + 1)

        self.clear_pathfinding_cache()

    def check_game_over(self):
        if self.hostages_rescued >= 7 or self.hostages_lost >= 4 or self.structural_damage >= 25:
            self.running = False

    def step(self):
        t = self.turn_counter + 1
        self.schedule.step()
        self.advance_disturbances()
        self.maintain_minimum_markers()
        self.check_game_over()
        self.datacollector.collect(self)
        self.logger.snapshot_tick(
            self, t=t, include_pois=False, include_riots=True, include_doors=False)
        self.turn_counter = t

    def reveal_if_needed(self, pos):
        if pos in self.revealed_pois:
            return False
        contents = self.get_contents_at(pos)
        kind = None
        if any(isinstance(c, Hostage) for c in contents):
            kind = "v"
        elif any(isinstance(c, FalseAlarm) for c in contents):
            kind = "f"
        if kind is None:
            return False

        self.revealed_pois.add(pos)
        if hasattr(self, "logger"):
            self.logger.reveal_poi(
                pos[1] + 1, pos[0] + 1, kind, t=self.turn_counter + 1)
        return True
