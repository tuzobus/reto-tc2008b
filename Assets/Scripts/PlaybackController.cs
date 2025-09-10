using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlaybackController : MonoBehaviour
{
    public JsonFileProvider provider;
    public GridBuilder grid;

    [Header("Prefabs")]
    public GameObject agentPrefab;
    public GameObject poiUnknownPrefab, poiVictimPrefab, poiFalsePrefab;
    public GameObject riotPrefab;
    public Transform agentsParent, iconsParent;

    [Header("Playback")]
    public float stepDuration = 0.5f;   // tiempo entre ticks t y t+1
    public float moveLerpTime = 0.25f;  // duración de cada movimiento
    public bool  debugLog = true;       // imprime lo que ejecuta

    [Header("Icon decals")]
    public float iconYOffset = 0.02f;
    public float iconFill    = 0.9f;
    public bool  iconPrefabIsPlane = true;

    MapConfig config;
    SimLog log;

    readonly Dictionary<string, GameObject> agents = new();
    readonly Dictionary<string, GameObject> icons  = new(); // key "type:r,c"

    void Start()
    {
        StartCoroutine(LoadAndPlay());
    }

    IEnumerator LoadAndPlay()
    {
        // 1) Cargar config
        yield return provider.LoadConfig(cfg => config = cfg);
        if (config == null)
        {
            Debug.LogError("Config no cargada. Revisa ruta y JSON.");
            yield break;
        }

        // 2) Cargar log
        yield return provider.LoadLog(lg => log = lg);
        if (log == null)
        {
            Debug.LogError("Log no cargado. Revisa ruta y JSON.");
            yield break;
        }

        // 3) Construir tablero
        BuildBoard(config);
        SeedInitialState(config);

        if (log.steps == null || log.steps.Count == 0)
        {
            Debug.LogWarning("Log sin 'steps'. Nada que reproducir.");
            yield break;
        }

        // 4) Reproducción: AGRUPADA POR T
        yield return PlayStepsGrouped(log.steps);

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
        if (cellRows == null || cellRows.Length != rows)
        {
            Debug.LogError("cellRows ausente o con conteo incorrecto.");
            return null;
        }

        var outCells = new string[rows][];
        for (int r = 0; r < rows; r++)
        {
            var parts = cellRows[r].Split(' ');
            if (parts.Length != cols)
            {
                Debug.LogError($"cellRows[{r}] no tiene {cols} columnas. Tiene {parts.Length}.");
                return null;
            }
            outCells[r] = parts;
        }
        return outCells;
    }

    void SeedInitialState(MapConfig c)
    {
        // POIs desconocidos al inicio
        foreach (var p in c.pois)
            PlaceOrSwapIcon($"poi:{p.r},{p.c}", poiUnknownPrefab, p.r, p.c);

        // Disturbios iniciales
        if (c.riots != null)
            foreach (var r in c.riots)
                PlaceOrSwapIcon($"riot:{r.r},{r.c}", riotPrefab, r.r, r.c);
    }

    // ===========================
    //      REPRODUCCIÓN AGRUPADA
    // ===========================
    IEnumerator PlayStepsGrouped(List<Step> steps)
    {
        // Asumimos que vienen por orden t ascendente (si no, ordenamos).
        steps.Sort((a,b) => a.t.CompareTo(b.t));

        int i = 0;
        while (i < steps.Count)
        {
            int t = steps[i].t;
            // recolecta el batch con el mismo t
            List<Step> batch = new();
            while (i < steps.Count && steps[i].t == t)
            {
                batch.Add(steps[i]);
                i++;
            }

            if (debugLog) Debug.Log($"=== TICK {t}  (steps={batch.Count}) ===");

            // 1) spawns primero (para que existan agentes antes de moverse)
            foreach (var s in batch)
                if (s.type == "spawn_agent") HandleSpawn(s);

            // 2) “instantáneos” SIN movimiento
            foreach (var s in batch)
            {
                switch (s.type)
                {
                    case "reveal_poi": HandleRevealPoi(s); break;
                    case "rescue":     HandleRescue(s);    break;
                    case "riot_spread":HandleRiotSpread(s);break;
                    case "riot_contained": HandleRiotContained(s); break;
                    case "damage_inc": /* actualiza UI si tienes */ break;
                    // move queda fuera; se hace después
                }
            }

            // 3) MOVER EN PARALELO
            //   Lanza todas las corutinas de move y espera a que todas terminen.
            int activeMoves = 0;
            List<Coroutine> running = new();

            foreach (var s in batch)
            {
                if (s.type == "move")
                {
                    if (agents.TryGetValue(s.id, out var go))
                    {
                        Vector3 target = grid.CellToWorld(s.to.r, s.to.c);
                        activeMoves++;
                        running.Add(StartCoroutine(LerpMove(go, target, moveLerpTime, () => { activeMoves--; })));
                        if (debugLog) Debug.Log($"t={t} MOVE {s.id}: ({s.from.r},{s.from.c})->({s.to.r},{s.to.c})");
                    }
                    else
                    {
                        Debug.LogWarning($"[Playback] t={t} MOVE: agente {s.id} no existe.");
                    }
                }
            }

            // Espera a que terminen todos los move de este tick
            while (activeMoves > 0) yield return null;

            // 4) Pausa entre ticks (ritmo de reproducción)
            if (stepDuration > 0f) yield return new WaitForSeconds(stepDuration);
        }
    }

    // =============== HANDLERS ===============
    void HandleSpawn(Step s)
    {
        if (!agents.ContainsKey(s.id))
        {
            var go = Instantiate(agentPrefab, grid.CellToWorld(s.r, s.c), Quaternion.identity, agentsParent);
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

        // Si tienes AgentView para giro/animación:
        var view = go.GetComponent<AgentView>();

        while (t < time)
        {
            t += Time.deltaTime;
            float a = Mathf.Clamp01(t / time);
            Vector3 next = Vector3.Lerp(start, target, a);
            if (view != null)
            {
                view.FaceDirection(next - go.transform.position);
                view.SetSpeed(1f);
            }
            go.transform.position = next;
            yield return null;
        }
        go.transform.position = target;
        if (view != null) view.SetSpeed(0f);

        onDone?.Invoke();
    }

    void PlaceOrSwapIcon(string key, GameObject prefab, int r, int c)
    {
        RemoveIcon(key);

        Vector3 pos = grid.CellToWorld(r, c);
        pos.y += iconYOffset;

        float s = iconPrefabIsPlane ? (grid.cellSize * iconFill / 10f) : (grid.cellSize * iconFill);
        var go  = Instantiate(prefab, pos, Quaternion.identity, iconsParent);
        go.transform.localScale = new Vector3(s, 1f, s);

        icons[key] = go;
    }

    void RemoveIcon(string key)
    {
        if (icons.TryGetValue(key, out var go))
        {
            Destroy(go);
            icons.Remove(key);
        }
    }
}
