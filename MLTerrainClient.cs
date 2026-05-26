using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using Random = UnityEngine.Random;
#if UNITY_EDITOR
using UnityEditor;
#endif

// Attach this script to the Terrain object.
public class MLTerrainClient : MonoBehaviour
{
    private struct SemanticWeights
    {
        public float water;
        public float forest;
        public float rock;
        public float land;
    }

    [Header("Inference")]
    public Texture2D inputMask;
    [SerializeField] private string serverUrl = "http://127.0.0.1:5000/generate";

    [Header("Terrain Shape")]
    [SerializeField] private float terrainWidth = 256f;
    [SerializeField] private float terrainLength = 256f;
    [SerializeField] private float terrainHeight = 34f;
    [SerializeField] private int semanticSmoothPasses = 2;
    [Range(0f, 1f)] [SerializeField] private float plainsSmoothing = 0.24f;
    [Range(0f, 1f)] [SerializeField] private float shorelineSmoothing = 0.36f;

    [Header("Terrain Layers")]
    [SerializeField] private TerrainLayer grassLayer;
    [SerializeField] private TerrainLayer sandLayer;
    [SerializeField] private TerrainLayer rockLayer;

    [Header("Object Prefabs")]
    public GameObject treePrefab;
    public GameObject rockPrefab;

    [Header("Tree Placement")]
    [SerializeField] private float treeCellSize = 6f;
    [Range(0f, 1f)] [SerializeField] private float treeDensity = 0.82f;
    [SerializeField] private float maxTreeSlope = 28f;
    [SerializeField] private Vector2 treeScaleRange = new Vector2(0.85f, 1.3f);

    [Header("Rock Placement")]
    [SerializeField] private float rockCellSize = 12f;
    [Range(0f, 1f)] [SerializeField] private float rockDensity = 0.28f;
    [SerializeField] private float minRockSlope = 14f;
    [SerializeField] private Vector2 rockScaleRange = new Vector2(0.8f, 1.45f);

    private const int Resolution = 256;
    private readonly List<GameObject> generatedObjects = new List<GameObject>();

    private void Start()
    {
        Debug.Log("<b>[ML Client]</b> Starting landscape request...");

        if (inputMask == null)
        {
            Debug.LogError("<b>[ML Client]</b> Input Mask is missing.");
            return;
        }

        StartCoroutine(RequestLandscapeGeneration());
    }

    private IEnumerator RequestLandscapeGeneration()
    {
        byte[] maskBytes;

        try
        {
            maskBytes = inputMask.EncodeToPNG();
        }
        catch (System.Exception exception)
        {
            Debug.LogError($"<b>[ML Client]</b> Failed to encode mask: {exception.Message}");
            yield break;
        }

        if (maskBytes == null || maskBytes.Length == 0)
        {
            Debug.LogError("<b>[ML Client]</b> Mask PNG payload is empty. Set texture compression to None and enable Read/Write.");
            yield break;
        }

        WWWForm form = new WWWForm();
        form.AddBinaryData("mask", maskBytes, "mask.png", "image/png");

        using (UnityWebRequest request = UnityWebRequest.Post(serverUrl, form))
        {
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"<b>[ML Client]</b> Server error: {request.error}");
                yield break;
            }

            ApplyHeightsToTerrain(request.downloadHandler.data);
        }
    }

    private void ApplyHeightsToTerrain(byte[] rawBytes)
    {
        if (rawBytes == null || rawBytes.Length != Resolution * Resolution * sizeof(float))
        {
            Debug.LogError($"<b>[ML Client]</b> Unexpected payload size: {rawBytes?.Length ?? 0} bytes.");
            return;
        }

        Terrain terrain = GetComponent<Terrain>();
        if (terrain == null || terrain.terrainData == null)
        {
            Debug.LogError("<b>[ML Client]</b> Terrain component was not found.");
            return;
        }

        float[,] heights = DecodeHeightmap(rawBytes);
        heights = SmoothHeightsBySemantics(heights);

        TerrainData terrainData = terrain.terrainData;
        terrainData.heightmapResolution = Resolution + 1;
        terrainData.size = new Vector3(terrainWidth, terrainHeight, terrainLength);
        terrainData.SetHeights(0, 0, heights);

        Debug.Log("<b>[ML Client]</b> Terrain heights updated.");

        AutoPaintTerrain(terrainData);
        PopulateTerrainObjects(terrain, terrainData);
    }

    private float[,] DecodeHeightmap(byte[] rawBytes)
    {
        float[,] heights = new float[Resolution, Resolution];

        for (int y = 0; y < Resolution; y++)
        {
            for (int x = 0; x < Resolution; x++)
            {
                int byteIndex = (y * Resolution + x) * sizeof(float);
                float height = BitConverter.ToSingle(rawBytes, byteIndex);
                heights[Resolution - 1 - y, x] = Mathf.Clamp01(height);
            }
        }

        return heights;
    }

    private float[,] SmoothHeightsBySemantics(float[,] source)
    {
        float[,] working = source;

        for (int pass = 0; pass < semanticSmoothPasses; pass++)
        {
            float[,] blurred = BoxBlur3x3(working);
            float[,] next = new float[Resolution, Resolution];

            for (int y = 0; y < Resolution; y++)
            {
                for (int x = 0; x < Resolution; x++)
                {
                    float normX = x / (float)(Resolution - 1);
                    float normY = y / (float)(Resolution - 1);

                    SemanticWeights weights = SampleSemanticWeights(normX, normY);
                    float nearbyWater = SampleAverageWater(normX, normY);
                    float shore = Mathf.Clamp01(nearbyWater - weights.water);
                    float plains = Mathf.Clamp01(weights.land + (weights.forest * 0.55f) - (weights.rock * 0.75f) - (weights.water * 0.6f));
                    float smoothStrength = (plains * plainsSmoothing) + (shore * shorelineSmoothing);

                    next[y, x] = Mathf.Lerp(working[y, x], blurred[y, x], Mathf.Clamp01(smoothStrength));
                }
            }

            working = next;
        }

        return working;
    }

    private float[,] BoxBlur3x3(float[,] source)
    {
        float[,] blurred = new float[Resolution, Resolution];

        for (int y = 0; y < Resolution; y++)
        {
            for (int x = 0; x < Resolution; x++)
            {
                float sum = 0f;
                int count = 0;

                for (int offsetY = -1; offsetY <= 1; offsetY++)
                {
                    int sampleY = Mathf.Clamp(y + offsetY, 0, Resolution - 1);

                    for (int offsetX = -1; offsetX <= 1; offsetX++)
                    {
                        int sampleX = Mathf.Clamp(x + offsetX, 0, Resolution - 1);
                        sum += source[sampleY, sampleX];
                        count++;
                    }
                }

                blurred[y, x] = sum / count;
            }
        }

        return blurred;
    }

    private void AutoPaintTerrain(TerrainData terrainData)
    {
        if (!EnsureTerrainLayers(terrainData))
        {
            Debug.LogWarning("<b>[ML Client]</b> Terrain painting skipped because Terrain Layers are not configured.");
            return;
        }

        int alphaResolution = terrainData.alphamapResolution;
        if (alphaResolution <= 0)
        {
            Debug.LogWarning("<b>[ML Client]</b> Terrain painting skipped because alphamap resolution is invalid.");
            return;
        }

        float[,,] splatmap = new float[alphaResolution, alphaResolution, 3];

        for (int y = 0; y < alphaResolution; y++)
        {
            for (int x = 0; x < alphaResolution; x++)
            {
                float normX = x / (float)(alphaResolution - 1);
                float normY = y / (float)(alphaResolution - 1);

                SemanticWeights weights = SampleSemanticWeights(normX, normY);
                float nearbyWater = SampleAverageWater(normX, normY);
                float shore = Mathf.Clamp01(nearbyWater - weights.water);
                float slope = terrainData.GetSteepness(normX, normY) / 90f;
                float height01 = terrainData.GetInterpolatedHeight(normX, normY) / Mathf.Max(terrainData.size.y, 0.001f);

                float grassWeight = Mathf.Clamp01(0.75f + (weights.forest * 0.45f) + (weights.land * 0.2f) - (slope * 1.25f) - (weights.rock * 0.6f) - (shore * 0.45f));
                float sandWeight = Mathf.Clamp01((weights.water * 1.4f) + (shore * 1.3f) + Mathf.Max(0f, 0.22f - height01) * 0.35f - (slope * 0.35f));
                float rockWeight = Mathf.Clamp01((weights.rock * 1.5f) + (slope * 1.25f) + Mathf.Max(0f, height01 - 0.55f) * 0.75f);

                float total = grassWeight + sandWeight + rockWeight;
                if (total <= 0.001f)
                {
                    grassWeight = 1f;
                    total = 1f;
                }

                splatmap[y, x, 0] = grassWeight / total;
                splatmap[y, x, 1] = sandWeight / total;
                splatmap[y, x, 2] = rockWeight / total;
            }
        }

        terrainData.SetAlphamaps(0, 0, splatmap);
        Debug.Log("<b>[ML Client]</b> Terrain textures updated.");
    }

    private bool EnsureTerrainLayers(TerrainData terrainData)
    {
        if (terrainData == null)
        {
            return false;
        }

        TerrainLayer[] currentLayers = terrainData.terrainLayers;
        if (currentLayers != null && currentLayers.Length >= 3)
        {
            return true;
        }

        TryAutoAssignTerrainLayers();

        if (grassLayer == null || sandLayer == null || rockLayer == null)
        {
            return false;
        }

        terrainData.terrainLayers = new TerrainLayer[] { grassLayer, sandLayer, rockLayer };
        return terrainData.terrainLayers != null && terrainData.terrainLayers.Length >= 3;
    }

    private void TryAutoAssignTerrainLayers()
    {
        if (grassLayer != null && sandLayer != null && rockLayer != null)
        {
            return;
        }

#if UNITY_EDITOR
        if (grassLayer == null)
        {
            grassLayer = AssetDatabase.LoadAssetAtPath<TerrainLayer>("Assets/Layer_Grass.terrainlayer");
        }

        if (sandLayer == null)
        {
            sandLayer = AssetDatabase.LoadAssetAtPath<TerrainLayer>("Assets/Layer_Sand.terrainlayer");
        }

        if (rockLayer == null)
        {
            rockLayer = AssetDatabase.LoadAssetAtPath<TerrainLayer>("Assets/Layer_Rock.terrainlayer");
        }
#endif
    }

    private void PopulateTerrainObjects(Terrain terrain, TerrainData terrainData)
    {
        ClearGeneratedObjects();

        if (treePrefab != null)
        {
            SpawnSemanticObjects(terrain, terrainData, treePrefab, treeCellSize, treeDensity, true);
        }

        if (rockPrefab != null)
        {
            SpawnSemanticObjects(terrain, terrainData, rockPrefab, rockCellSize, rockDensity, false);
        }

        Debug.Log($"<b>[ML Client]</b> Object placement finished. Spawned {generatedObjects.Count} objects.");
    }

    private void SpawnSemanticObjects(Terrain terrain, TerrainData terrainData, GameObject prefab, float cellSize, float density, bool isTree)
    {
        if (prefab == null || cellSize <= 0.01f)
        {
            return;
        }

        int cellsX = Mathf.Max(1, Mathf.CeilToInt(terrainData.size.x / cellSize));
        int cellsZ = Mathf.Max(1, Mathf.CeilToInt(terrainData.size.z / cellSize));
        float noiseOffsetX = Random.Range(0f, 1000f);
        float noiseOffsetY = Random.Range(0f, 1000f);

        for (int cellZ = 0; cellZ < cellsZ; cellZ++)
        {
            for (int cellX = 0; cellX < cellsX; cellX++)
            {
                float baseNormX = Mathf.Clamp01((cellX + 0.5f) / cellsX);
                float baseNormY = Mathf.Clamp01((cellZ + 0.5f) / cellsZ);
                float nearbyForest = isTree ? SampleAverageForest(baseNormX, baseNormY) : 0f;
                int attempts = 1;

                if (isTree)
                {
                    attempts = nearbyForest > 0.58f ? 3 : (nearbyForest > 0.28f ? 2 : 1);
                }

                for (int attempt = 0; attempt < attempts; attempt++)
                {
                    float normX = Mathf.Clamp01((cellX + Random.value) / cellsX);
                    float normY = Mathf.Clamp01((cellZ + Random.value) / cellsZ);

                    SemanticWeights weights = SampleSemanticWeights(normX, normY);
                    if (weights.water > 0.2f)
                    {
                        continue;
                    }

                    float slope = terrainData.GetSteepness(normX, normY);
                    float height01 = terrainData.GetInterpolatedHeight(normX, normY) / Mathf.Max(terrainData.size.y, 0.001f);
                    float cluster = Mathf.PerlinNoise((normX * 4.8f) + noiseOffsetX, (normY * 4.8f) + noiseOffsetY);

                    float spawnChance;
                    Quaternion rotation;
                    Vector2 scaleRange;

                    if (isTree)
                    {
                        if (slope > maxTreeSlope)
                        {
                            continue;
                        }

                        float localForest = SampleAverageForest(normX, normY);
                        float treeSuitability = Mathf.Clamp01((weights.forest * 1.4f) + (localForest * 0.9f) + (weights.land * 0.18f) - (weights.rock * 0.35f));
                        spawnChance = treeSuitability * density * Mathf.Lerp(0.9f, 1.28f, cluster) * Mathf.Clamp01(1f - (slope / maxTreeSlope));
                        spawnChance *= Mathf.Lerp(0.78f, 1f, Mathf.Clamp01((height01 - 0.04f) / 0.45f));
                        rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
                        scaleRange = treeScaleRange;
                    }
                    else
                    {
                        float steepnessBias = Mathf.Clamp01((slope - minRockSlope) / 30f);
                        float rockSuitability = Mathf.Clamp01((weights.rock * 1.05f) + steepnessBias + Mathf.Max(0f, height01 - 0.42f) * 0.32f);
                        spawnChance = rockSuitability * density * Mathf.Lerp(0.55f, 0.92f, cluster);

                        Vector3 normal = terrainData.GetInterpolatedNormal(normX, normY);
                        rotation = Quaternion.FromToRotation(Vector3.up, normal) * Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
                        scaleRange = rockScaleRange;
                    }

                    if (Random.value > Mathf.Clamp01(spawnChance))
                    {
                        continue;
                    }

                    Vector3 localPosition = new Vector3(normX * terrainData.size.x, 0f, normY * terrainData.size.z);
                    float localHeight = terrain.SampleHeight(terrain.transform.position + localPosition);
                    Vector3 worldPosition = terrain.transform.position + new Vector3(localPosition.x, localHeight, localPosition.z);

                    GameObject instance = Instantiate(prefab, worldPosition, rotation);
                    float uniformScale = Random.Range(scaleRange.x, scaleRange.y);
                    instance.transform.localScale = Vector3.Scale(instance.transform.localScale, Vector3.one * uniformScale);
                    generatedObjects.Add(instance);
                }
            }
        }
    }

    private void ClearGeneratedObjects()
    {
        foreach (GameObject instance in generatedObjects)
        {
            if (instance != null)
            {
                Destroy(instance);
            }
        }

        generatedObjects.Clear();
    }

    private SemanticWeights SampleSemanticWeights(float normX, float normY)
    {
        Color pixel = inputMask.GetPixelBilinear(normX, normY);

        float water = Mathf.Clamp01(pixel.b - (Mathf.Max(pixel.r, pixel.g) * 0.55f));
        float forest = Mathf.Clamp01(pixel.r - (Mathf.Max(pixel.g, pixel.b) * 0.55f));
        float rock = Mathf.Clamp01(pixel.g - (Mathf.Max(pixel.r, pixel.b) * 0.55f));
        float land = Mathf.Clamp01(1f - Mathf.Max(water, Mathf.Max(forest, rock)));

        return new SemanticWeights
        {
            water = water,
            forest = forest,
            rock = rock,
            land = land
        };
    }

    private float SampleAverageWater(float normX, float normY)
    {
        if (inputMask == null)
        {
            return 0f;
        }

        float texelX = 3f / inputMask.width;
        float texelY = 3f / inputMask.height;
        float total = 0f;
        float weightSum = 0f;

        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            for (int offsetX = -1; offsetX <= 1; offsetX++)
            {
                float sampleX = Mathf.Clamp01(normX + (offsetX * texelX));
                float sampleY = Mathf.Clamp01(normY + (offsetY * texelY));
                float weight = (offsetX == 0 && offsetY == 0) ? 2f : 1f;

                total += SampleSemanticWeights(sampleX, sampleY).water * weight;
                weightSum += weight;
            }
        }

        return total / Mathf.Max(weightSum, 0.001f);
    }

    private float SampleAverageForest(float normX, float normY)
    {
        if (inputMask == null)
        {
            return 0f;
        }

        float texelX = 4f / inputMask.width;
        float texelY = 4f / inputMask.height;
        float total = 0f;
        float weightSum = 0f;

        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            for (int offsetX = -1; offsetX <= 1; offsetX++)
            {
                float sampleX = Mathf.Clamp01(normX + (offsetX * texelX));
                float sampleY = Mathf.Clamp01(normY + (offsetY * texelY));
                float weight = (offsetX == 0 && offsetY == 0) ? 2f : 1f;

                total += SampleSemanticWeights(sampleX, sampleY).forest * weight;
                weightSum += weight;
            }
        }

        return total / Mathf.Max(weightSum, 0.001f);
    }
}
