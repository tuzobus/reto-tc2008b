using UnityEngine;
using System.Collections.Generic;

public class GridBuilder : MonoBehaviour
{
    public enum TileType { Plane10x10, Quad1x1, GenericFlat }

    [Header("Grid")]
    public float cellSize = 1f;

    [Header("Parents")]
    public Transform tilesParent, wallsParent, doorsParent, entriesParent;

    [Header("Prefabs")]
    public GameObject tilePrefab, wallPrefab, doorPrefab, entryPrefab;

    [Header("Tile visuals")]
    public TileType tileType = TileType.Plane10x10;
    [Range(0.1f, 1f)] public float tileFill = 1f;
    public float tileYOffset = 0f;
    public float genericTileThickness = 0.02f;

    [Header("Door visuals")]
    public Material doorClosedMat;
    public Material doorOpenMat;

    [Header("Entry decals (centradas en la celda)")]
    public float entryYOffset = 0.03f;
    public float entryFill    = 0.7f;
    public bool entryPrefabIsPlane = true;

    // Puertas para actualizaciones
    readonly Dictionary<string, DoorVisual> doorMap = new();

    // ========================
    //     CELDA a MUNDO(1-based)
    // ========================
    public Vector3 CenterOfCell(int r, int c)
    {
        float x = (c - 1) * cellSize;
        float z = - (r - 1) * cellSize;
        return new Vector3(x, 0f, z);
    }

    public (int r, int c) WorldToCell(Vector3 world)
    {
        int c = Mathf.RoundToInt((world.x / cellSize) + 1f);
        int r = Mathf.RoundToInt((-world.z / cellSize) + 1f);
        return (r, c);
    }

    public Vector3 SnapToCellCenter(Vector3 world) {
        var (r,c) = WorldToCell(world);
        return CenterOfCell(r,c);
    }

    // ========================
    //     CONSTRUCCIÓN
    // ========================
    public void BuildTiles(int rows, int cols)
    {
        ClearParent(tilesParent);

        if (tilesParent != null && tilesParent.lossyScale != Vector3.one)
            Debug.LogWarning("[GridBuilder] tilesParent tiene escala != 1. Recomiendo (1,1,1).");

        for (int r = 1; r <= rows; r++)
        for (int c = 1; c <= cols; c++)
        {
            var pos = CenterOfCell(r, c) + Vector3.up * tileYOffset;
            var go  = Instantiate(tilePrefab, pos, Quaternion.identity, tilesParent);

            switch (tileType)
            {
                case TileType.Plane10x10:
                {
                    float s = cellSize * tileFill / 10f;
                    go.transform.localScale = new Vector3(s, 1f, s);
                    break;
                }
                case TileType.Quad1x1:
                {
                    float s = cellSize * tileFill;
                    go.transform.localScale = new Vector3(s, 1f, s);
                    break;
                }
                case TileType.GenericFlat:
                {
                    float sx = cellSize * tileFill;
                    float sz = cellSize * tileFill;
                    float sy = genericTileThickness;
                    go.transform.localScale = new Vector3(sx, sy, sz);

                    go.transform.position += Vector3.up * (sy * 0.5f);
                    break;
                }
            }
        }
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
            string code = cells[r - 1][c - 1];
            var basePos = CenterOfCell(r, c);
            if (string.IsNullOrEmpty(code) || code.Length != 4) continue;

            if (code[0] == '1') PlaceWall(basePos, 0f);
            if (code[1] == '1') PlaceWall(basePos, 90f);
            if (code[2] == '1') PlaceWall(basePos, 180f);
            if (code[3] == '1') PlaceWall(basePos, 270f);
        }
    }

    void PlaceWall(Vector3 cellCenter, float rotY)
    {
        float half = cellSize * 0.5f;
        Vector3 offset = rotY switch
        {
            0f   => new Vector3(0, 0,  half),
            90f  => new Vector3(-half, 0, 0),
            180f => new Vector3(0, 0, -half),
            270f => new Vector3( half, 0, 0),
            _    => Vector3.zero
        };

        var go = Instantiate(wallPrefab, cellCenter + offset, Quaternion.Euler(0, rotY, 0), wallsParent);
        // Asegura largo = cellSize en el eje de la pared
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
            if (d.r1 == d.r2 && Mathf.Abs(d.c1 - d.c2) == 1)
            {
                int r = d.r1;
                int cMin = Mathf.Min(d.c1, d.c2);
                Vector3 cell = CenterOfCell(r, cMin);

                Vector3 pos = cell + new Vector3(+half, 0f, 0f);
                float rotY = 90f;

                var go = Instantiate(doorPrefab, pos, Quaternion.Euler(0, rotY, 0), doorsParent);
                go.transform.localScale = new Vector3(cellSize * 0.6f, go.transform.localScale.y, go.transform.localScale.z);
                go.transform.position  += Vector3.up * (go.transform.localScale.y * 0.5f);

                RegisterDoor(go, d);
            }
            else if (d.c1 == d.c2 && Mathf.Abs(d.r1 - d.r2) == 1)
            {
                int c = d.c1;
                int rMin = Mathf.Min(d.r1, d.r2);
                Vector3 cell = CenterOfCell(rMin, c);

                Vector3 pos = cell + new Vector3(0f, 0f, -half);
                float rotY = 0f;

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

        doorMap[DoorKey(d.r1, d.c1, d.r2, d.c2)] = vis;
        doorMap[DoorKey(d.r2, d.c2, d.r1, d.c1)] = vis;
    }

    public void SetDoorStateByCells(int r1, int c1, int r2, int c2, bool isOpen)
    {
        var key = DoorKey(r1, c1, r2, c2);
        if (doorMap.TryGetValue(key, out var vis)) vis.SetOpen(isOpen);
        else Debug.LogWarning($"[GridBuilder] No encontré puerta {key} para actualizar estado.");
    }

    static string DoorKey(int r1, int c1, int r2, int c2)
    {
        if (r2 < r1 || (r2 == r1 && c2 < c1)) { (r1, r2) = (r2, r1); (c1, c2) = (c2, c1); }
        return $"{r1},{c1}-{r2},{c2}";
    }

    public void BuildEntryMarkers(List<RC> entries)
    {
        ClearParent(entriesParent);
        if (entries == null || entryPrefab == null) return;

        foreach (var e in entries)
        {
            Vector3 pos = CenterOfCell(e.r, e.c);
            pos.y += entryYOffset;

            float s = entryPrefabIsPlane ? (cellSize * entryFill / 10f) : (cellSize * entryFill);
            var go  = Instantiate(entryPrefab, pos, Quaternion.identity, entriesParent);
            go.transform.localScale = new Vector3(s, 1f, s);

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

public class DoorVisual : MonoBehaviour
{
    Renderer[] rends;
    Material closedMat, openMat;
    bool isOpen;

    public void Init(Material closed, Material open, bool initialOpen)
    {
        closedMat = closed; openMat = open;
        rends = GetComponentsInChildren<Renderer>(true);
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
