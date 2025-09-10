using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlaybackController : MonoBehaviour
{
    public enum PlaybackMode { Snapshots, EventsCompressed, EventsStepByStep }

    public JsonFileProvider provider;
    public GridBuilder grid;

    [Header("Prefabs")]
    public GameObject agentPrefab;
    public GameObject poiUnknownPrefab, poiVictimPrefab, poiFalsePrefab;
    public GameObject riotPrefab;
    public Transform agentsParent, iconsParent;

    [Header("Playback")]
    public PlaybackMode mode = PlaybackMode.Snapshots;
    [Tooltip("Pausa entre ticks (solo rige el ritmo global; no afecta la duración de cada Lerp).")]
    public float stepDuration = 0.5f;
    [Tooltip("Duración del movimiento cuando hay animación (compressed o step-by-step).")]
    public float moveLerpTime = 0.25f;
    public bool  debugLog = true;

    [Header("Icon decals")]
    public float iconYOffset = 0.02f;
    public float iconFill    = 0.9f;
    public bool  iconPrefabIsPlane = true;

    MapConfig config;
    SimLog log;

    readonly Dictionary<string, GameObject> agents = new();
    readonly Dictionary<string, GameObject> icons  = new();

    // snapshots indexados por t
    readonly Dictionary<int, Snapshot> snapshotsByT = new();

    void Start()
    {
        StartCoroutine(LoadAndPlay());
    }

    void Update()
    {
        // Atajos para alternar en vivo
        if (Input.GetKeyDown(KeyCode.Alpha1)) { mode = PlaybackMode.Snapshots; if (debugLog) Debug.Log("Modo → Snapshots"); }
        if (Input.GetKeyDown(KeyCode.Alpha2)) { mode = PlaybackMode.EventsCompressed; if (debugLog) Debug.Log("Modo → Eventos (comprimido)"); }
        if (Input.GetKeyDown(KeyCode.Alpha3)) { mode = PlaybackMode.EventsStepByStep; if (debugLog) Debug.Log("Modo → Eventos (paso a paso)"); }
    }

    IEnumerator LoadAndPlay()
    {
        // 1) Cargar config
        yield return provider.LoadConfig(cfg => config = cfg);
        if (config == null) { Debug.LogError("Config no cargada."); yield break; }

        // 2) Cargar log
        yield return provider.LoadLog(lg => log = lg);
        if (log == null) { Debug.LogError("Log no cargado."); yield break; }

        // 3) Indexar snapshots (si los hay)
        snapshotsByT.Clear();
        if (log.snapshots != null)
            foreach (var s in log.snapshots) snapshotsByT[s.t] = s;

        // 4) Construir tablero + estado inicial
        BuildBoard(config);
        SeedInitialState(config);

        // 5) Reproducir según modo
        switch (mode)
        {
            case PlaybackMode.Snapshots:
                if (log.snapshots == null || log.snapshots.Count == 0)
                {
                    Debug.LogWarning("Modo Snapshots seleccionado, pero no hay snapshots en el log. Cambiando a Eventos (comprimido).");
                    mode = PlaybackMode.EventsCompressed;
                    yield return PlayByEventsCompressed(log.steps);
                }
                else
                {
                    yield return PlayBySnapshots(); // usa snapshots (t=0..N consecutivo)
                }
                break;

            case PlaybackMode.EventsCompressed:
                yield return PlayByEventsCompressed(log.steps);
                break;

            case PlaybackMode.EventsStepByStep:
                yield return PlayByEventsStepByStep(log.steps);
                break;
        }

        if (debugLog)
            Debug.Log($"Resultado: {log.result}  Rescued:{log.rescued} Lost:{log.lost} Damage:{log.damage}");
    }

    void BuildBoard(MapConfig c)
    {
        var parsedCells = c.cells ?? ParseCellsFromRows(c.cellRows, c.rows, c.cols);
        grid.BuildTiles(c.rows, c.cols);
        grid.BuildWalls(parsedCells);
        grid.BuildDoors(c.doors);
        grid.BuildEntryMarkers(c.entries);
    }

    static string[][] ParseCellsFromRows(string[] cellRows, int rows, int cols)
    {
        if (cellRows == null || cellRows.Length != rows) return null;
        var outCells = new string[rows][];
        for (int r = 0; r < rows; r++) outCells[r] = cellRows[r].Split(' ');
        return outCells;
    }

    void SeedInitialState(MapConfig c)
    {
        // POIs desconocidos
        if (c.pois != null)
            foreach (var p in c.pois)
                PlaceOrSwapIcon($"poi:{p.r},{p.c}", poiUnknownPrefab, p.r, p.c);

        // Disturbios iniciales
        if (c.riots != null)
            foreach (var r in c.riots)
                PlaceOrSwapIcon($"riot:{r.r},{r.c}", riotPrefab, r.r, r.c);
    }

    // ===========================
    //      MODO: SNAPSHOTS
    // ===========================
    IEnumerator PlayBySnapshots()
    {
        // asumimos t=0..T consecutivo en snapshots
        int t = 0;
        while (snapshotsByT.ContainsKey(t))
        {
            ApplySnapshotAtTick(t);
            if (debugLog) Debug.Log($"[Snapshot] aplicado t={t}");
            yield return new WaitForSeconds(stepDuration);
            t++;
        }
    }

    void ApplySnapshotAtTick(int t)
    {
        if (!snapshotsByT.TryGetValue(t, out var snap) || snap == null) return;

        foreach (var a in snap.agents)
        {
            if (!agents.TryGetValue(a.id, out var go))
            {
                go = Instantiate(agentPrefab, grid.CenterOfCell(a.r, a.c), Quaternion.identity, agentsParent);
                agents[a.id] = go;
            }
            else
            {
                // Teletransporte exacto al snapshot
                agents[a.id].transform.position = grid.CenterOfCell(a.r, a.c);
            }
        }
    }

    // ===========================
    //      MODO: EVENTOS (COMPRIMIDO)
    // ===========================
    IEnumerator PlayByEventsCompressed(List<Step> steps)
    {
        if (steps == null || steps.Count == 0) yield break;
        steps.Sort((a,b) => a.t.CompareTo(b.t));

        int i = 0;
        while (i < steps.Count)
        {
            int t = steps[i].t;

            // batch por tick (nota: si no hay eventos, ese t no existe aquí: saltos de t son normales)
            List<Step> batch = new();
            while (i < steps.Count && steps[i].t == t) { batch.Add(steps[i]); i++; }

            if (debugLog) Debug.Log($"=== TICK {t}  (steps={batch.Count}) ===");

            // spawns primero
            foreach (var s in batch) if (s.type == "spawn_agent") HandleSpawn(s);

            // eventos sin movimiento
            foreach (var s in batch)
            {
                switch (s.type)
                {
                    case "reveal_poi": HandleRevealPoi(s); break;
                    case "rescue":     HandleRescue(s);    break;
                    case "riot_spread":HandleRiotSpread(s);break;
                    case "riot_contained": HandleRiotContained(s); break;
                    case "damage_inc": break;
                }
            }

            // mover (solo último destino por agente en este tick)
            var lastMove = new Dictionary<string, Step>();
            foreach (var s in batch) if (s.type == "move") lastMove[s.id] = s;

            int active = 0;
            foreach (var kv in lastMove)
            {
                var s = kv.Value;
                if (!agents.TryGetValue(s.id, out var go)) continue;

                var (rNow, cNow) = grid.WorldToCell(go.transform.position);
                if (rNow == s.to.r && cNow == s.to.c) continue; // ya está ahí

                Vector3 target = grid.CenterOfCell(s.to.r, s.to.c);
                active++;
                StartCoroutine(LerpMove(go, target, moveLerpTime, () => active--));

                if (debugLog) Debug.Log($"t={t} MOVE (compressed) {s.id}: ->({s.to.r},{s.to.c})");
            }

            while (active > 0) yield return null;
            if (stepDuration > 0f) yield return new WaitForSeconds(stepDuration);
        }
    }

    // ===========================
    //      MODO: EVENTOS (PASO A PASO)
    // ===========================
    IEnumerator PlayByEventsStepByStep(List<Step> steps)
    {
        if (steps == null || steps.Count == 0) yield break;
        steps.Sort((a,b) => a.t.CompareTo(b.t));

        int i = 0;
        while (i < steps.Count)
        {
            int t = steps[i].t;
            List<Step> batch = new();
            while (i < steps.Count && steps[i].t == t) { batch.Add(steps[i]); i++; }

            if (debugLog) Debug.Log($"=== TICK {t}  (steps={batch.Count}) ===");

            // spawns
            foreach (var s in batch) if (s.type == "spawn_agent") HandleSpawn(s);

            // eventos sin movimiento
            foreach (var s in batch)
            {
                switch (s.type)
                {
                    case "reveal_poi": HandleRevealPoi(s); break;
                    case "rescue":     HandleRescue(s);    break;
                    case "riot_spread":HandleRiotSpread(s);break;
                    case "riot_contained": HandleRiotContained(s); break;
                    case "damage_inc": break;
                }
            }

            // agrupar TODOS los moves por agente y reproducirlos en orden
            var movesByAgent = new Dictionary<string, List<Step>>();
            foreach (var s in batch)
                if (s.type == "move")
                {
                    if (!movesByAgent.TryGetValue(s.id, out var list))
                        list = movesByAgent[s.id] = new List<Step>();
                    list.Add(s);
                }

            int active = 0;
            foreach (var kv in movesByAgent)
            {
                if (!agents.TryGetValue(kv.Key, out var go)) continue;
                active++;
                StartCoroutine(PlayAgentMoves(go, kv.Value, moveLerpTime, () => active--));
            }

            while (active > 0) yield return null;
            if (stepDuration > 0f) yield return new WaitForSeconds(stepDuration);
        }
    }

    IEnumerator PlayAgentMoves(GameObject go, List<Step> moves, float totalTime, System.Action onDone)
    {
        float perMove = Mathf.Max(0.01f, totalTime / Mathf.Max(1, moves.Count));
        foreach (var m in moves)
        {
            Vector3 target = grid.CenterOfCell(m.to.r, m.to.c);
            float t = 0f; Vector3 start = go.transform.position;
            while (t < perMove)
            {
                t += Time.deltaTime;
                go.transform.position = Vector3.Lerp(start, target, Mathf.Clamp01(t / perMove));
                yield return null;
            }
            go.transform.position = target;
        }
        onDone?.Invoke();
    }

    // =============== HANDLERS ===============
    void HandleSpawn(Step s)
    {
        if (!agents.ContainsKey(s.id))
        {
            var go = Instantiate(agentPrefab, grid.CenterOfCell(s.r, s.c), Quaternion.identity, agentsParent);
            agents[s.id] = go;
            if (debugLog) Debug.Log($"SPAWN {s.id} at ({s.r},{s.c})");
        }
    }

    void HandleRevealPoi(Step s)
    {
        RemoveIcon($"poi:{s.r},{s.c}");
        var prefab = s.kind == "v" ? poiVictimPrefab : poiFalsePrefab;
        PlaceOrSwapIcon($"poi:{s.r},{s.c}", prefab, s.r, s.c);
        if (debugLog) Debug.Log($"REVEAL_POI ({s.r},{s.c}) kind={s.kind}");
    }

    void HandleRescue(Step s)
    {
        RemoveIcon($"poi:{s.r},{s.c}");
        if (debugLog) Debug.Log($"RESCUE at ({s.r},{s.c})");
    }

    void HandleRiotSpread(Step s)
    {
        PlaceOrSwapIcon($"riot:{s.to.r},{s.to.c}", riotPrefab, s.to.r, s.to.c);
        if (debugLog) Debug.Log($"RIOT_SPREAD {s.from.r},{s.from.c} -> {s.to.r},{s.to.c}");
    }

    void HandleRiotContained(Step s)
    {
        RemoveIcon($"riot:{s.r},{s.c}");
        if (debugLog) Debug.Log($"RIOT_CONTAINED at ({s.r},{s.c})");
    }

    // ============== UTILIDADES ==============
    IEnumerator LerpMove(GameObject go, Vector3 target, float time, System.Action onDone)
    {
        Vector3 start = go.transform.position;
        float t = 0f;
        while (t < time)
        {
            t += Time.deltaTime;
            go.transform.position = Vector3.Lerp(start, target, Mathf.Clamp01(t/time));
            yield return null;
        }
        go.transform.position = target;
        onDone?.Invoke();
    }

    void PlaceOrSwapIcon(string key, GameObject prefab, int r, int c)
    {
        RemoveIcon(key);
        Vector3 pos = grid.CenterOfCell(r, c);
        pos.y += iconYOffset;
        float s = iconPrefabIsPlane ? (grid.cellSize * iconFill / 10f) : (grid.cellSize * iconFill);
        var go  = Instantiate(prefab, pos, Quaternion.identity, iconsParent);
        go.transform.localScale = new Vector3(s, 1f, s);
        icons[key] = go;
    }

    void RemoveIcon(string key)
    {
        if (icons.TryGetValue(key, out var go)) { Destroy(go); icons.Remove(key); }
    }
}
