from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

# logger simple para eventos + snapshots (r,c en base 1 para unity)


class SimLogger:
    def __init__(self):
        self.steps = []
        self.snapshots = []

    def spawn_agent(self, aid, r, c, t):
        self.steps.append({"t": t, "type": "spawn_agent",
                           "id": str(aid), "r": r, "c": c})

    def move(self, aid, from_pos, to_pos, t):
        # mesa usa (x,y) base 0; unity espera (r,c) base 1
        self.steps.append({
            "t": t, "type": "move", "id": str(aid),
            "from": {"r": from_pos[1] + 1, "c": from_pos[0] + 1},
            "to":   {"r": to_pos[1] + 1,   "c": to_pos[0] + 1}
        })

    def reveal_poi(self, r, c, kind, t):
        self.steps.append({"t": t, "type": "reveal_poi",
                           "r": r, "c": c, "kind": kind})

    def rescue(self, r, c, t):
        self.steps.append({"t": t, "type": "rescue", "r": r, "c": c})

    def riot_spread(self, from_pos, to_pos, t):
        self.steps.append({
            "t": t, "type": "riot_spread",
            "from": {"r": from_pos[1] + 1, "c": from_pos[0] + 1},
            "to":   {"r": to_pos[1] + 1,   "c": to_pos[0] + 1}
        })

    def riot_contained(self, r, c, t):
        self.steps.append({"t": t, "type": "riot_contained", "r": r, "c": c})

    def damage_inc(self, amount, t):
        self.steps.append(
            {"t": t, "type": "damage_inc", "amount": int(amount)})

    def snapshot_tick(self, model, t, include_pois=False, include_riots=False, include_doors=False):
        snap = {"t": int(t)}

        agents = []
        for a in model.schedule.agents:
            if getattr(a, "pos", None) is not None:
                agents.append(
                    {"id": str(a.unique_id), "r": a.pos[1] + 1, "c": a.pos[0] + 1})
        agents.sort(key=lambda x: int(x["id"]))
        snap["agents"] = agents

        if include_pois:
            pois = []
            for (x, y), contents in model.cell_contents.items():
                if any(isinstance(c, Hostage) for c in contents):
                    pois.append({"r": y + 1, "c": x + 1, "kind": "v"})
                elif any(isinstance(c, FalseAlarm) for c in contents):
                    pois.append({"r": y + 1, "c": x + 1, "kind": "f"})
            snap["pois"] = sorted(pois, key=lambda p: (p["r"], p["c"]))

        if include_riots:
            riots = []
            for (x, y), contents in model.cell_contents.items():
                d = next((c for c in contents if isinstance(c, Disturbance)), None)
                if d:
                    riots.append(
                        {"r": y + 1, "c": x + 1, "severity": d.severity})
            snap["riots"] = sorted(riots, key=lambda p: (p["r"], p["c"]))

        if include_doors:
            doors = []
            for (x, y), contents in model.cell_contents.items():
                for g in contents:
                    if isinstance(g, Gate):
                        doors.append(
                            {"r": y + 1, "c": x + 1, "open": bool(g.is_open)})
            snap["doors"] = sorted(doors, key=lambda d: (d["r"], d["c"]))

        self.snapshots.append(snap)

    def to_simlog(self, result, rescued, lost, damage, meta=None):
        out = {
            "result": result,
            "rescued": rescued,
            "lost": lost,
            "damage": damage,
            "steps": self.steps,
            "snapshots": self.snapshots
        }
        if meta:
            out["meta"] = meta
        return out


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
        self.severity = severity  # mild/active/grave
        self.turns_in_current_state = 0


def get_grid(model):
    # matriz para ver muros y cosas en 2x resolución (solo debug/plots)
    H = model.grid.height
    W = model.grid.width
    canvas = np.zeros((H * 2 + 1, W * 2 + 1), dtype=np.int32)

    for (x, y), walls in model.walls.items():
        cy, cx = y * 2 + 1, x * 2 + 1
        if walls.get('top', False):
            canvas[cy - 1, cx] = 1
        if walls.get('bottom', False):
            canvas[cy + 1, cx] = 1
        if walls.get('left', False):
            canvas[cy, cx - 1] = 1
        if walls.get('right', False):
            canvas[cy, cx + 1] = 1

    canvas[0, :] = canvas[-1, :] = 1
    canvas[:, 0] = canvas[:, -1] = 1

    for (x, y), contents in model.cell_contents.items():
        cy, cx = y * 2 + 1, x * 2 + 1
        if any(isinstance(c, Hostage) for c in contents):
            canvas[cy, cx] = 3
            continue
        if any(isinstance(c, Disturbance) for c in contents):
            d = next(c for c in contents if isinstance(c, Disturbance))
            canvas[cy, cx] = 8 if d.severity == 'grave' else (
                5 if d.severity == 'active' else 4)
            continue
        if any(isinstance(c, FalseAlarm) for c in contents):
            canvas[cy, cx] = 7
            continue
        if any(isinstance(c, Gate) for c in contents):
            g = next(c for c in contents if isinstance(c, Gate))
            canvas[cy, cx] = 9 if g.is_open else 10
            continue

    for (ex, ey) in getattr(model, "entry_points", []):
        cy, cx = ey * 2 + 1, ex * 2 + 1
        if canvas[cy, cx] == 0:
            canvas[cy, cx] = 6

    for agent in model.schedule.agents:
        if getattr(agent, "pos", None) is None:
            continue
        x, y = agent.pos
        cy, cx = y * 2 + 1, x * 2 + 1
        canvas[cy, cx] = 2

    return canvas


def get_grid_board(model):
    """
    matriz HxW (6x8) con contenido por celda. codigos rapidos:
      0 vacio, 2 agente, 3 rehen,
      4/5/8 disturbio leve/activo/grave,
      7 falsa, 6 entrada, 9/10 reja abierta/cerrada
      (no dibuja muros aqui)
    """
    H, W = model.grid.height, model.grid.width
    M = np.zeros((H, W), dtype=np.int32)

    for (ex, ey) in getattr(model, "entry_points", []):
        if M[ey, ex] == 0:
            M[ey, ex] = 6

    for (x, y), contents in model.cell_contents.items():
        if any(isinstance(c, Hostage) for c in contents):
            M[y, x] = 3
            continue
        if any(isinstance(c, Disturbance) for c in contents):
            d = next(c for c in contents if isinstance(c, Disturbance))
            M[y, x] = 8 if d.severity == "grave" else (
                5 if d.severity == "active" else 4)
            continue
        if any(isinstance(c, FalseAlarm) for c in contents):
            M[y, x] = 7
            continue
        if any(isinstance(c, Gate) for c in contents):
            g = next(c for c in contents if isinstance(c, Gate))
            M[y, x] = 9 if g.is_open else 10
            continue

    for a in model.schedule.agents:
        if getattr(a, "pos", None) is None:
            continue
        x, y = a.pos
        M[y, x] = 2

    return M


class TacticalAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        self.action_points = 4
        self.carrying_hostage = False

    def step(self):
        # cada tick reinicia a 4 ap
        self.action_points = 4

        while self.action_points > 0:
            possible = []
            here = self.model.get_contents_at(self.pos)

            # movimiento ortogonal (muros/puertas cuentan)
            for nb in self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False):
                if self.model.can_move_to(self.pos, nb):
                    cost = 1
                    nb_cont = self.model.get_contents_at(nb)
                    if any(isinstance(c, Disturbance) for c in nb_cont):
                        cost = 2
                    if self.action_points >= cost:
                        possible.append(("move", nb, cost))

            # rescate (2 ap) solo si no llevas rehén
            if not self.carrying_hostage:
                h = next((c for c in here if isinstance(c, Hostage)), None)
                if h and self.action_points >= 2:
                    possible.append(("rescue", h, 2))

            # investigar falsa (1 ap)
            fa = next((c for c in here if isinstance(c, FalseAlarm)), None)
            if fa and self.action_points >= 1:
                possible.append(("investigate", fa, 1))

            # dejar rehén en entrada (1 ap)
            if self.carrying_hostage and self.pos in self.model.entry_points and self.action_points >= 1:
                possible.append(("dropoff", None, 1))

            # contener disturbio (segun severidad)
            d = next((c for c in here if isinstance(c, Disturbance)), None)
            if d:
                if d.severity == "mild" and self.action_points >= 1:
                    possible.append(("contain", d, 1))
                elif d.severity == "active" and self.action_points >= 2:
                    possible.append(("contain", d, 2))

            if not possible:
                break

            action, target, cost = self.random.choice(possible)

            if action == "move":
                from_pos = self.pos
                self.model.grid.move_agent(self, target)
                if hasattr(self.model, "logger"):
                    self.model.logger.move(
                        self.unique_id, from_pos, target, t=self.model.turn_counter + 1)

                # al entrar a celda, revelar poi si aplica
                revealed_now = self.model.reveal_if_needed(self.pos)
                # si se reveló, cortamos turno para separar ticks en el log
                if revealed_now:
                    self.action_points = 0
                    continue

            elif action == "rescue":
                # asegurar que ya este revelado antes de quitar icono
                self.model.reveal_if_needed(self.pos)
                if hasattr(self.model, "logger"):
                    self.model.logger.rescue(
                        self.pos[1] + 1, self.pos[0] + 1, t=self.model.turn_counter + 1)
                self.carrying_hostage = True
                self.model.remove_entity(target, self.pos)

            elif action == "investigate":
                self.model.reveal_if_needed(self.pos)
                self.model.false_alarms_investigated += 1
                self.model.remove_entity(target, self.pos)

            elif action == "dropoff":
                self.carrying_hostage = False
                self.model.hostages_rescued += 1
                # no se emite 'rescue' aqui; el icono ya se quito al pickup

            elif action == "contain":
                self.model.remove_entity(target, self.pos)
                if hasattr(self.model, "logger"):
                    self.model.logger.riot_contained(
                        self.pos[1] + 1, self.pos[0] + 1, t=self.model.turn_counter + 1)

            self.action_points -= cost


class RescueModel(Model):
    def __init__(self, config_path="config.json"):
        super().__init__()
        cfg = self._load_config(config_path)

        rows = cfg["rows"]     # alto
        cols = cfg["cols"]     # ancho

        # ojo: mesa usa (width=cols, height=rows)
        self.grid = MultiGrid(cols, rows, torus=False)
        self.schedule = RandomActivation(self)
        self.cell_contents = defaultdict(list)
        self.running = True

        # stats
        self.hostages_rescued = 0
        self.hostages_lost = 0
        self.structural_damage = 0
        self.false_alarms_investigated = 0
        self.next_entity_id = 0
        self.turn_counter = 0
        self.min_hidden_markers = 3

        # desde config
        self.walls = {}            # {(x,y): {top,left,bottom,right}}
        self.entry_points = []     # [(x,y)]
        self._build_from_config(cfg)

        self.logger = SimLogger()
        self.revealed_pois = set()

        # spawns iniciales (6) en entradas
        for _ in range(6):
            a = TacticalAgent(self.get_next_id(), self)
            self.schedule.add(a)
            ep = self.random.choice(self.entry_points)
            self.grid.place_agent(a, ep)

        for a in self.schedule.agents:
            if getattr(a, "pos", None) is not None:
                r, c = a.pos[1] + 1, a.pos[0] + 1
                self.logger.spawn_agent(a.unique_id, r, c, t=0)

        self.initial_hostages = sum(1 for cont in self.cell_contents.values(
        ) for it in cont if isinstance(it, Hostage))
        self.initial_alarms = sum(1 for cont in self.cell_contents.values(
        ) for it in cont if isinstance(it, FalseAlarm))
        self.initial_disturbances = sum(1 for cont in self.cell_contents.values(
        ) for it in cont if isinstance(it, Disturbance))

        self.datacollector = DataCollector(
            model_reporters={"Grid": lambda m: np.array(get_grid_board(m))})

        # snapshot inicial (t=0) para alinear con unity
        self.logger.snapshot_tick(
            self, t=0, include_pois=False, include_riots=False, include_doors=False)

    # parseo de config
    def _load_config(self, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_cells(self, cfg):
        if "cells" in cfg and cfg["cells"]:
            return cfg["cells"]
        # respaldo si viene en filas de strings "abcd ..."
        rows = cfg["rows"]
        cols = cfg["cols"]
        out = []
        for line in cfg["cellRows"]:
            parts = line.strip().split()
            if len(parts) != cols:
                raise ValueError(
                    f"linea con {len(parts)} cols; se esperaban {cols}.")
            out.append(parts)
        if len(out) != rows:
            raise ValueError(
                f"se recibieron {len(out)} filas; se esperaban {rows}.")
        return out

    def _build_from_config(self, cfg):
        # paredes por celda, codigo "abcd" = up,left,down,right
        cells = self._parse_cells(cfg)
        rows, cols = cfg["rows"], cfg["cols"]
        for r in range(rows):
            for c in range(cols):
                code = cells[r][c]
                if len(code) != 4:
                    raise ValueError(
                        f"celda ({r},{c}) codigo invalido: {code}")
                x, y = c, r
                self.walls[(x, y)] = {
                    "top":    code[0] == "1",
                    "left":   code[1] == "1",
                    "bottom": code[2] == "1",
                    "right":  code[3] == "1",
                }

        # entradas (r,c) base 1: (x,y) base 0
        self.entry_points = []
        for e in cfg.get("entries", []):
            r, c = e["r"], e["c"]
            self.entry_points.append((c - 1, r - 1))

        # pois
        for p in cfg.get("pois", []):
            r, c, kind = p["r"], p["c"], p["kind"]
            pos = (c - 1, r - 1)
            if kind == "v":
                self.cell_contents[pos].append(Hostage(self.get_next_id()))
            else:
                self.cell_contents[pos].append(FalseAlarm(self.get_next_id()))

        # disturbios
        for rr in cfg.get("riots", []):
            r, c = rr["r"], rr["c"]
            pos = (c - 1, r - 1)
            self.cell_contents[pos].append(
                Disturbance(self.get_next_id(), "mild"))

        # puertas (simplificado: objeto en celda)
        for d in cfg.get("doors", []):
            r1, c1, r2, c2 = d["r1"], d["c1"], d["r2"], d["c2"]
            is_open = bool(d.get("open", False))
            pos1 = (c1 - 1, r1 - 1)
            self.cell_contents[pos1].append(Gate(self.get_next_id(), is_open))

    # utilidades
    def get_next_id(self):
        self.next_entity_id += 1
        return self.next_entity_id

    def get_contents_at(self, pos):
        return self.grid.get_cell_list_contents([pos]) + self.cell_contents.get(pos, [])

    def remove_entity(self, entity, pos):
        if pos in self.cell_contents and entity in self.cell_contents[pos]:
            self.cell_contents[pos].remove(entity)

    def can_move_to(self, from_pos, to_pos):
        x2, y2 = to_pos
        if not (0 <= x2 < self.grid.width and 0 <= y2 < self.grid.height):
            return False

        x1, y1 = from_pos
        dx, dy = x2 - x1, y2 - y1

        # checar pared en origen segun direccion
        w = self.walls.get(from_pos, {})
        if dx == 1 and w.get('right'):
            return False
        elif dx == -1 and w.get('left'):
            return False
        elif dy == 1 and w.get('bottom'):
            return False
        elif dy == -1 and w.get('top'):
            return False

        # puerta cerrada en destino bloquea
        contents = self.get_contents_at(to_pos)
        if any(isinstance(g, Gate) and not g.is_open for g in contents):
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
        # rompe en ambos lados si aplica
        x1, y1 = pos1
        x2, y2 = pos2
        dx, dy = x2 - x1, y2 - y1

        w1 = self.walls.get(pos1, {})
        w2 = self.walls.get(pos2, {})

        if dx == 1 and w1.get('right'):
            w1['right'] = False
            w2['left'] = False
        elif dx == -1 and w1.get('left'):
            w1['left'] = False
            w2['right'] = False
        elif dy == 1 and w1.get('bottom'):
            w1['bottom'] = False
            w2['top'] = False
        elif dy == -1 and w1.get('top'):
            w1['top'] = False
            w2['bottom'] = False

        self.walls[pos1] = w1
        self.walls[pos2] = w2

    def get_available_cell(self):
        # libre de agente y de gate cerrada
        for _ in range(200):
            pos = (self.random.randrange(self.grid.width),
                   self.random.randrange(self.grid.height))
            cont = self.get_contents_at(pos)
            has_agent = any(isinstance(c, TacticalAgent) for c in cont)
            has_closed_gate = any(isinstance(
                c, Gate) and not c.is_open for c in cont)
            if not (has_agent or has_closed_gate):
                return pos
        return (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height))

    def place_passive_entity(self, entity_class):
        entity = entity_class(self.get_next_id(
        ), 'mild') if entity_class == Disturbance else entity_class(self.get_next_id())
        pos = self.get_available_cell()
        self.cell_contents[pos].append(entity)

    def count_hidden_markers(self):
        return sum(1 for cont in self.cell_contents.values() for it in cont if isinstance(it, (Hostage, FalseAlarm)))

    def maintain_minimum_markers(self):
        while self.count_hidden_markers() < self.min_hidden_markers:
            cls = self.random.choices(
                [Hostage, FalseAlarm], weights=[0.7, 0.3], k=1)[0]
            self.place_passive_entity(cls)

    def advance_disturbances(self):
        # progresion simple por turnos; activa: grave puede explotar
        for pos, cont in list(self.cell_contents.items()):
            ds = [d for d in cont if isinstance(d, Disturbance)]
            for d in ds:
                d.turns_in_current_state += 1
                if d.severity == "mild" and d.turns_in_current_state >= 4:
                    d.severity = "active"
                    d.turns_in_current_state = 0
                elif d.severity == "active" and d.turns_in_current_state >= 6:
                    d.severity = "grave"
                    self.handle_explosion(pos, cont)

        # chance baja de nuevo disturbio
        if self.random.random() < 0.05:
            pos = self.get_available_cell()
            cont = self.get_contents_at(pos)
            if not any(isinstance(c, Disturbance) for c in cont):
                self.cell_contents[pos].append(
                    Disturbance(self.get_next_id(), "mild"))

    def handle_explosion(self, pos, contents):
        self.structural_damage += 1
        if hasattr(self, "logger"):
            self.logger.damage_inc(1, t=self.turn_counter + 1)

        neighbors = [(pos[0] + dx, pos[1] + dy)
                     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        for nb in neighbors:
            if self.has_wall_between(pos, nb) and self.random.random() < 0.3:
                self.break_wall_between(pos, nb)

        # puertas cercanas pueden romperse
        for nb in neighbors:
            if nb in self.cell_contents:
                gs = [g for g in self.cell_contents[nb] if isinstance(g, Gate)]
                for g in gs:
                    if self.random.random() < 0.5:
                        self.remove_entity(g, nb)

        # rehenes perdidos en la celda
        for h in [h for h in contents if isinstance(h, Hostage)]:
            self.hostages_lost += 1
            self.remove_entity(h, pos)

        # el disturbio desaparece tras la explosion
        for d in [d for d in contents if isinstance(d, Disturbance)]:
            self.remove_entity(d, pos)

    def check_game_over(self):
        if self.hostages_rescued >= 7 or self.hostages_lost >= 4 or self.structural_damage >= 25:
            self.running = False

    def step(self):
        # t que se resuelve (1..n)
        t = self.turn_counter + 1

        # primero agentes (emiten eventos con t actual)
        self.schedule.step()

        # luego sistema
        self.advance_disturbances()
        self.maintain_minimum_markers()
        self.check_game_over()

        # colecta para plots/anim
        self.datacollector.collect(self)

        # snapshot final del tick (lo que unity deberia ver en frame t)
        self.logger.snapshot_tick(
            self, t=t, include_pois=False, include_riots=False, include_doors=False)

        self.turn_counter = t

    def reveal_if_needed(self, pos):
        """
        si hay host/falsa en pos y no estaba revelado, emite reveal y marca.
        devuelve True si revelo ahora.
        """
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
