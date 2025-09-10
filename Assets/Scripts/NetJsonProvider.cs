using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class NetJsonProvider : MonoBehaviour
{
    [Header("Server")]
    public string baseUrl = "http://127.0.0.1:5000";

    public IEnumerator LoadConfig(Action<MapConfig> onDone, Action<string> onError = null)
    {
        using var req = UnityWebRequest.Get($"{baseUrl}/config");
        yield return req.SendWebRequest();
        if (req.result != UnityWebRequest.Result.Success) { onError?.Invoke(req.error); yield break; }
        var cfg = JsonUtility.FromJson<MapConfig>(req.downloadHandler.text);
        if (cfg == null) { onError?.Invoke("Config parse returned null"); yield break; }
        onDone?.Invoke(cfg);
    }

    public IEnumerator LoadFullLog(Action<SimLog> onDone, Action<string> onError = null, int? maxSteps = null, int? seed = null)
    {
        string url = $"{baseUrl}/run";
        bool hasQ = false;
        if (maxSteps.HasValue) { url += $"{(hasQ?"&":"?")}max_steps={maxSteps.Value}"; hasQ = true; }
        if (seed.HasValue)     { url += $"{(hasQ?"&":"?")}seed={seed.Value}"; hasQ = true; }

        using var req = UnityWebRequest.Get(url);
        yield return req.SendWebRequest();
        if (req.result != UnityWebRequest.Result.Success) { onError?.Invoke(req.error); yield break; }

        var log = JsonUtility.FromJson<SimLog>(req.downloadHandler.text);
        if (log == null) { onError?.Invoke("SimLog parse returned null"); yield break; }
        onDone?.Invoke(log);
    }
}
