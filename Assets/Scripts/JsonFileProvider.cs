using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

public class JsonFileProvider : MonoBehaviour
{
    public string configFile = "config.json";
    public string logFile    = "log.json";

    public IEnumerator LoadConfig(System.Action<MapConfig> onDone)
    {
        string path = Path.Combine(Application.streamingAssetsPath, configFile);
        yield return LoadJson<MapConfig>(path, onDone);
    }

    public IEnumerator LoadLog(System.Action<SimLog> onDone)
    {
        string path = Path.Combine(Application.streamingAssetsPath, logFile);
        yield return LoadJson<SimLog>(path, onDone);
    }

    IEnumerator LoadJson<T>(string path, System.Action<T> onDone)
    {
        string url = path;
        if (!url.StartsWith("http") && !url.StartsWith("file://"))
            url = "file://" + url;

        Debug.Log($"[JsonFileProvider] Loading: {url}");

        using (UnityWebRequest req = UnityWebRequest.Get(url))
        {
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Load error: " + req.error);
                yield break;
            }

            string json = req.downloadHandler.text;
            if (string.IsNullOrWhiteSpace(json))
            {
                Debug.LogError("Load error: empty JSON payload");
                yield break;
            }

            T obj;
            try
            {
                obj = JsonUtility.FromJson<T>(json);
            }
            catch (System.Exception ex)
            {
                Debug.LogError("JSON parse error: " + ex.Message);
                yield break;
            }

            if (obj == null)
            {
                Debug.LogError("JSON parse returned null (check JSON structure matches DataModels).");
                yield break;
            }

            onDone?.Invoke(obj);
        }
    }
}
