using System;
using System.Collections.Generic;

[Serializable] public class Poi  { public int r, c; public string kind; }
[Serializable] public class RC   { public int r, c; }
[Serializable] public class Door { public int r1, c1, r2, c2; public bool open; } // 'open' opcional

// Opcional: si algún día el config trae severidad inicial de disturbios
[Serializable] public class Riot { public int r, c; public string severity; }

[Serializable]
public class MapConfig {
    public int rows, cols;
    public string[][] cells;     // o...
    public string[]   cellRows;  // respaldo: líneas "abcd abcd ..."
    public List<Poi>  pois;
    public List<RC>   riots;            // compatible con tu JSON actual
    public List<Riot> riotsDetailed;    // opcional, si algún día quieren severidad inicial
    public List<Door> doors;
    public List<RC>   entries;
}

[Serializable]
public class Step {
    public int    t;
    public string type;
    public string id;

    public RC from, to;   // para 'move' y 'riot_spread'
    public int r, c;      // coords directas
    public string kind;   // para 'reveal_poi'
    public int amount;    // para 'damage_inc'

    // —— Nuevos campos flexibles ——
    public string severity; // "mild"/"active"/"grave" (para riot_set/riot_upgrade)
    public bool   open;     // estado de puerta (para door_state/open_door/close_door)
    public int    r1, c1, r2, c2; // identificar una puerta por las dos celdas vecinas
}

[Serializable]
public class SimLog {
    public List<Step> steps;
    public string result;
    public int rescued, lost, damage;
}
