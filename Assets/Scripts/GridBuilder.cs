using UnityEngine;
using System.Collections.Generic;

public class GridBuilder : MonoBehaviour
{
    public float cellSize = 1f;

    public Transform tilesParent, wallsParent, doorsParent, entriesParent;
    public GameObject tilePrefab, wallPrefab, doorPrefab, entryPrefab;

    [Header("Door visuals")]
    public Material doorClosedMat;
    public Material doorOpenMat;

    [Header("Entry decals (centradas en la celda)")]
    public float entryYOffset = 0.03f;    // levanta un poco (evitar z-fighting)
    public float entryFill    = 0.7f;     // 0..1 del tamaño de la celda
    public bool  entryPrefabIsPlane = true; // Plane=10x10; Quad=1x1 (girado X=90 en el prefab)

    // Mapa de puertas para poder actualizar su estado en runtime
    readonly Dictionary<string, DoorVisual> doorMap = new();

    public Vector3 CellToWorld(int r, int c)
    {
        float x = (c - 1) * cellSize;
        float z = - (r - 1) * cellSize;
        return new Vector3(x, 0f, z);
    }

    public void BuildTiles(int rows, int cols)
    {
        ClearParent(tilesParent);
        for (int r = 1; r <= rows; r++)
        for (int c = 1; c <= cols; c++)
            Instantiate(tilePrefab, CellToWorld(r, c), Quaternion.identity, tilesParent);
    }

    public void BuildWalls(string[][] cells)
    {
        ClearParent(wallsParent);
        if (cells == null || cells.Length == 0)
        {
            Debug.LogError("cells es null o vacío; no se pueden construir muros.");
            return;
        }

        int rows = cells.Length;
        int cols = cells[0].Length;

        for (int r = 1; r <= rows; r++)
        for (int c = 1; c <= cols; c++)
        {
            string code = cells[r - 1][c - 1]; // "abcd" = up,left,down,right
            var basePos = CellToWorld(r, c);
            if (string.IsNullOrEmpty(code) || code.Length != 4) continue;

            if (code[0] == '1') PlaceWall(basePos, 0f);   // up
            if (code[1] == '1') PlaceWall(basePos, 90f);  // left
            if (code[2] == '1') PlaceWall(basePos, 180f); // down
            if (code[3] == '1') PlaceWall(basePos, 270f); // right
        }
    }

    void PlaceWall(Vector3 cellCenter, float rotY)
    {
        float half = cellSize * 0.5f;
        Vector3 offset = rotY switch
        {
            0f   => new Vector3(0, 0,  half),  // borde superior
            90f  => new Vector3(-half, 0, 0),  // borde izquierdo
            180f => new Vector3(0, 0, -half),  // borde inferior
            270f => new Vector3( half, 0, 0),  // borde derecho
            _    => Vector3.zero
        };

        var go = Instantiate(wallPrefab, cellCenter + offset, Quaternion.Euler(0, rotY, 0), wallsParent);
        go.transform.localScale = new Vector3(cellSize, go.transform.localScale.y, go.transform.localScale.z);
        go.transform.position  += Vector3.up * (go.transform.localScale.y * 0.5f);
    }

    public void BuildDoors(List<Door> doors)
    {
        ClearParent(doorsParent);
        doorMap.Clear();
        if (doors == null) return;

        float half = cellSize * 0.5f;

        foreach (var d in doors)
        {
            // Horizontal: (r, c) ↔ (r, c+1)
            if (d.r1 == d.r2 && Mathf.Abs(d.c1 - d.c2) == 1)
            {
                int r = d.r1;
                int cMin = Mathf.Min(d.c1, d.c2);
                Vector3 cell = CellToWorld(r, cMin);

                Vector3 pos = cell + new Vector3(+half, 0f, 0f); // borde derecho de (r,cMin)
                float rotY = 90f; // a lo largo del eje Z

                var go = Instantiate(doorPrefab, pos, Quaternion.Euler(0, rotY, 0), doorsParent);
                go.transform.localScale = new Vector3(cellSize * 0.6f, go.transform.localScale.y, go.transform.localScale.z);
                go.transform.position  += Vector3.up * (go.transform.localScale.y * 0.5f);

                RegisterDoor(go, d);
            }
            // Vertical: (r, c) ↔ (r+1, c)
            else if (d.c1 == d.c2 && Mathf.Abs(d.r1 - d.r2) == 1)
            {
                int c = d.c1;
                int rMin = Mathf.Min(d.r1, d.r2);
                Vector3 cell = CellToWorld(rMin, c);

                Vector3 pos = cell + new Vector3(0f, 0f, -half); // borde inferior de (rMin,c)
                float rotY = 0f; // a lo largo del eje X

                var go = Instantiate(doorPrefab, pos, Quaternion.Euler(0, rotY, 0), doorsParent);
                go.transform.localScale = new Vector3(cellSize * 0.6f, go.transform.localScale.y, go.transform.localScale.z);
                go.transform.position  += Vector3.up * (go.transform.localScale.y * 0.5f);

                RegisterDoor(go, d);
            }
            else
            {
                Debug.LogWarning($"Puerta no adyacente: ({d.r1},{d.c1})-({d.r2},{d.c2})");
            }
        }
    }

    void RegisterDoor(GameObject go, Door d)
    {
        var vis = go.GetComponent<DoorVisual>();
        if (vis == null) vis = go.AddComponent<DoorVisual>();
        vis.Init(doorClosedMat, doorOpenMat, d.open);

        doorMap[DoorKey(d.r1, c1: d.c1, d.r2, d.c2)] = vis;
        doorMap[DoorKey(d.r2, c1: d.c2, d.r1, d.c1)] = vis; // clave simétrica
    }

    public void SetDoorStateByCells(int r1, int c1, int r2, int c2, bool isOpen)
    {
        var key = DoorKey(r1, c1, r2, c2);
        if (doorMap.TryGetValue(key, out var vis))
            vis.SetOpen(isOpen);
        else
            Debug.LogWarning($"[GridBuilder] No encontré puerta {key} para actualizar estado.");
    }

    static string DoorKey(int r1, int c1, int r2, int c2)
    {
        // Normaliza el par para que A-B == B-A
        if (r2 < r1 || (r2 == r1 && c2 < c1))
        {
            (r1, r2) = (r2, r1);
            (c1, c2) = (c2, c1);
        }
        return $"{r1},{c1}-{r2},{c2}";
    }

    // Entries como decals centrados en la celda (sin tocar muros)
    public void BuildEntryMarkers(List<RC> entries)
    {
        ClearParent(entriesParent);
        if (entries == null || entryPrefab == null) return;

        foreach (var e in entries)
        {
            Vector3 pos = CellToWorld(e.r, e.c);
            pos.y += entryYOffset;

            float s = entryPrefabIsPlane ? (cellSize * entryFill / 10f) : (cellSize * entryFill);
            var go  = Instantiate(entryPrefab, pos, Quaternion.identity, entriesParent);
            go.transform.localScale = new Vector3(s, 1f, s);

            // Si tu prefab es Quad (1x1), asegúrate de que esté ROTADO EN EL PREFAB a X=90.
            // (Si quieres forzarlo desde aquí, descomenta:)
            // go.transform.rotation = Quaternion.Euler(90f, 0f, 0f);

            var r = go.GetComponentInChildren<Renderer>();
            if (r != null)
            {
                r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                r.receiveShadows    = false;
            }
        }
    }

    void ClearParent(Transform t)
    {
        if (t == null) return;
        for (int i = t.childCount - 1; i >= 0; i--)
            DestroyImmediate(t.GetChild(i).gameObject);
    }
}

// Componente simple para cambiar material según estado (open/closed)
public class DoorVisual : MonoBehaviour
{
    Renderer[] rends;
    Material closedMat, openMat;
    bool isOpen;

    public void Init(Material closed, Material open, bool initialOpen)
    {
        closedMat = closed;
        openMat   = open;
        rends     = GetComponentsInChildren<Renderer>(true);
        SetOpen(initialOpen, true);
    }

    public void SetOpen(bool open, bool force = false)
    {
        if (!force && open == isOpen) return;
        isOpen = open;
        var mat = isOpen ? openMat : closedMat;
        if (mat == null || rends == null) return;
        foreach (var r in rends) r.sharedMaterial = mat;
    }
}
