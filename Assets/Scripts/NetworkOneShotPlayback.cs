using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NetworkOneShotPlayback : MonoBehaviour
{
    public NetJsonProvider provider;
    public GridBuilder grid;

    [Header("Icon decals")]
    public float iconYOffset = 0.02f;
    public float iconFill = 0.9f;
    public bool  iconPrefabIsPlane = true;

    [Header("Prefabs")]
    public GameObject agentPrefab;
    public GameObject poiUnknownPrefab, poiVictimPrefab, poiFalsePrefab;
    public GameObject riotPrefab;
    public Transform agentsParent, iconsParent;

    [Header("Playback")]
    public bool useSnapshotsStrict = true; // 1:1 con unity
    public bool eventsCompressed = false;  // si snapshots=false: último destino por tick
    public float stepDuration = 0.5f;
    public float moveLerpTime = 0.25f;
    public bool debugLog = true;

    MapConfig config;
    SimLog log;

    readonly Dictionary<string, GameObject> agents = new();
    readonly Dictionary<string, GameObject> icons  = new();
    readonly Dictionary<int, Snapshot> snapsByT = new();

    readonly List<int> orderedTs = new();

    void Start() { StartCoroutine(Run()); }

    IEnumerator Run()
    {
        // config
        yield return provider.LoadConfig(
            cfg => config = cfg,
            err => Debug.LogError($"[NetJsonProvider] {err}")
        );
        if (config == null) yield break;

        // descarga SimLog completo
        yield return provider.LoadFullLog(
            lg => log = lg,
            err => Debug.LogError($"[NetJsonProvider] {err}")
        );
        if (log == null) yield break;

        // construye tablero e iniciales
        var parsedCells = config.cells ?? ParseCellsFromRows(config.cellRows, config.rows, config.cols);
        grid.BuildTiles(config.rows, config.cols);
        grid.BuildWalls(parsedCells);
        grid.BuildDoors(config.doors);
        grid.BuildEntryMarkers(config.entries);
        SeedInitialIcons(config);

        snapsByT.Clear();
        if (log.snapshots != null)
            foreach (var s in log.snapshots) snapsByT[s.t] = s;

        orderedTs.Clear();
        orderedTs.AddRange(snapsByT.Keys);
        orderedTs.Sort();
        if (debugLog && orderedTs.Count > 0)
            Debug.Log($"[Net] Snapshots disponibles (primeros 12): {string.Join(",", orderedTs.GetRange(0, Mathf.Min(12, orderedTs.Count)))}");

        yield return PlayStepsGrouped(log.steps);

        if (debugLog)
            Debug.Log($"Resultado: {log.result}  Rescued:{log.rescued} Lost:{log.lost} Damage:{log.damage}");
    }

    static string[][] ParseCellsFromRows(string[] rows, int R, int C)
    {
        if (rows == null || rows.Length != R) return null;
        var outCells = new string[R][];
        for (int r = 0; r < R; r++) outCells[r] = rows[r].Split(' ');
        return outCells;
    }

    void SeedInitialIcons(MapConfig c)
    {
        if (c.pois != null)
            foreach (var p in c.pois) PlaceOrSwapIcon($"poi:{p.r},{p.c}", poiUnknownPrefab, p.r, p.c);
        if (c.riots != null)
            foreach (var rr in c.riots) PlaceOrSwapIcon($"riot:{rr.r},{rr.c}", riotPrefab, rr.r, rr.c);
    }

    IEnumerator PlayStepsGrouped(List<Step> steps)
    {
        if (steps != null) steps.Sort((a,b) => a.t.CompareTo(b.t));

        if (useSnapshotsStrict)
        {
            foreach (var t in orderedTs)
            {
                // aplica snapshot exacto
                if (snapsByT.TryGetValue(t, out var snap))
                {
                    ApplySnapshot(snap);
                    if (debugLog) Debug.Log($"[Snapshot] t={t}");
                }

                // eventos de ese mismo t (sin animar moves)
                List<Step> batch = new();
                if (steps != null)
                {
                    for (int k = 0; k < steps.Count; k++)
                        if (steps[k].t == t) batch.Add(steps[k]);
                }
                foreach (var s in batch)
                {
                    switch (s.type)
                    {
                        case "spawn_agent": break;
                        case "reveal_poi": HandleRevealPoi(s); break;
                        case "rescue":     HandleRescue(s);    break;
                        case "riot_spread":HandleRiotSpread(s);break;
                        case "riot_contained": HandleRiotContained(s); break;
                        case "damage_inc": break;
                        case "break_wall":
                            grid.RemoveWallByCells(s.r1, s.c1, s.r2, s.c2);
                            break;
                    }
                }

                if (stepDuration > 0) yield return new WaitForSeconds(stepDuration);
            }
            yield break;
        }

        // reproducción por eventos compressed / step-by-step
        int i = 0;
        while (i < (steps?.Count ?? 0))
        {
            int t = steps[i].t;
            List<Step> batch = new();
            while (i < steps.Count && steps[i].t == t) { batch.Add(steps[i]); i++; }

            // eventos sin movimiento
            foreach (var s in batch)
            {
                switch (s.type)
                {
                    case "spawn_agent": HandleSpawn(s); break;
                    case "reveal_poi": HandleRevealPoi(s); break;
                    case "rescue":     HandleRescue(s);    break;
                    case "riot_spread":HandleRiotSpread(s);break;
                    case "riot_contained": HandleRiotContained(s); break;
                    case "damage_inc": break;
                    case "break_wall":
                        grid.RemoveWallByCells(s.r1, s.c1, s.r2, s.c2);
                        break;
                }
            }

            if (eventsCompressed) yield return AnimateMovesCompressed(batch);
            else                  yield return AnimateMovesStepByStep(batch);

            if (stepDuration > 0) yield return new WaitForSeconds(stepDuration);
        }
    }

    void ApplySnapshot(Snapshot snap)
    {
        if (snap == null) return;

        // 1) Agentes (como ya lo haces)
        if (snap.agents != null)
        {
            foreach (var a in snap.agents)
            {
                if (!agents.TryGetValue(a.id, out var go))
                    agents[a.id] = Instantiate(agentPrefab, grid.CenterOfCell(a.r, a.c), Quaternion.identity, agentsParent);
                else
                    go.transform.position = grid.CenterOfCell(a.r, a.c);
            }
        }

        // 2) Disturbios: limpiar y volver a poner exactamente los del snapshot
        if (snap.riots != null)
        {
            // Elimina TODOS los íconos "riot:*"
            var toDelete = new List<string>();
            foreach (var kv in icons)
                if (kv.Key.StartsWith("riot:")) toDelete.Add(kv.Key);
            foreach (var k in toDelete) RemoveIcon(k);

            // Crea los que vengan en el snapshot
            foreach (var r in snap.riots)
                PlaceOrSwapIcon($"riot:{r.r},{r.c}", riotPrefab, r.r, r.c);
        }
    }

    IEnumerator AnimateMovesCompressed(List<Step> batch)
    {
        var last = new Dictionary<string, Step>();
        foreach (var s in batch) if (s.type=="move") last[s.id] = s;

        int active = 0;
        foreach (var kv in last)
        {
            var s = kv.Value;
            if (!agents.TryGetValue(s.id, out var go)) continue;
            Vector3 target = grid.CenterOfCell(s.to.r, s.to.c);
            active++;
            StartCoroutine(LerpMove(go, target, moveLerpTime, ()=>active--));
        }
        while (active>0) yield return null;
    }

    IEnumerator AnimateMovesStepByStep(List<Step> batch)
    {
        foreach (var s in batch)
        {
            if (s.type!="move") continue;
            if (!agents.TryGetValue(s.id, out var go)) continue;
            Vector3 target = grid.CenterOfCell(s.to.r, s.to.c);
            yield return LerpMove(go, target, moveLerpTime, null);
        }
    }

    void HandleSpawn(Step s)
    {
        if (!agents.ContainsKey(s.id))
        {
            var go = Instantiate(agentPrefab, grid.CenterOfCell(s.r, s.c), Quaternion.identity, agentsParent);
            agents[s.id] = go;
        }
    }
    void HandleRevealPoi(Step s)
    {
        RemoveIcon($"poi:{s.r},{s.c}");
        var prefab = s.kind == "v" ? poiVictimPrefab : poiFalsePrefab;
        PlaceOrSwapIcon($"poi:{s.r},{s.c}", prefab, s.r, s.c);
    }
    void HandleRescue(Step s) { RemoveIcon($"poi:{s.r},{s.c}"); }
    void HandleRiotSpread(Step s) { PlaceOrSwapIcon($"riot:{s.to.r},{s.to.c}", riotPrefab, s.to.r, s.to.c); }
    void HandleRiotContained(Step s) { RemoveIcon($"riot:{s.r},{s.c}"); }

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

        float s = iconPrefabIsPlane ? (grid.cellSize * iconFill / 10f)
                                    : (grid.cellSize * iconFill);
        var go  = Instantiate(prefab, pos, Quaternion.identity, iconsParent);
        go.transform.localScale = new Vector3(s, 1f, s);

        var rr = go.GetComponentInChildren<Renderer>();
        if (rr != null)
        {
            rr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            rr.receiveShadows    = false;
        }
        icons[key] = go;
    }

    void RemoveIcon(string key)
    {
        if (icons.TryGetValue(key, out var go)) { Destroy(go); icons.Remove(key); }
    }
}
