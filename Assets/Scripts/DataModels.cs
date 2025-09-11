using System;
using System.Collections.Generic;

[Serializable] public class Poi  { public int r, c; public string kind; }
[Serializable] public class RC   { public int r, c; }
[Serializable] public class Door { public int r1, c1, r2, c2; public bool open; }
[Serializable] public class Riot { public int r, c; public string severity; }

[Serializable]
public class MapConfig {
    public int rows, cols;
    public string[][] cells;
    public string[]   cellRows;
    public List<Poi>  pois;
    public List<RC>   riots;
    public List<Riot> riotsDetailed;
    public List<Door> doors;
    public List<RC>   entries;
}

[Serializable]
public class Step {
    public int    t;
    public string type;
    public string id;

    public RC from, to;
    public int r, c;
    public string kind;
    public int amount;

    public string severity; // "mild"/"active"/"grave"
    public bool   open;     // estado de puerta
    public int    r1, c1, r2, c2; // identificar puerta
}

[Serializable]
public class AgentState {
    public string id;
    public int r, c;
}

[Serializable]
public class Snapshot {
    public int t;
    public List<AgentState> agents;
    public List<Riot> riots;
}

[Serializable]
public class SimLog {
    public List<Step> steps;
    public List<Snapshot> snapshots;
    public string result;
    public int rescued, lost, damage;
}
