using System.Collections;
using System.IO;
using UnityEngine;

public class DatasetGenerator : MonoBehaviour
{
    [Header("Dataset")]
    public int datasetSize = 100;
    public string folderName = "ML_Dataset";

    [Header("Resolution")]
    public int resolution = 256;
    public float terrainHeight = 28f;

    [Header("Terrain Shape")]
    [Range(0.12f, 0.4f)] public float baseWaterLevel = 0.24f;
    [Range(0.6f, 3f)] public float macroScale = 1.4f;
    [Range(2f, 8f)] public float rollingScale = 4.8f;
    [Range(2f, 7f)] public float moistureScale = 3.1f;
    [Range(2f, 8f)] public float ridgeScale = 4.2f;
    public Vector2 riverWidthRange = new Vector2(0.016f, 0.032f);

    private string basePath;

    private void Start()
    {
        basePath = Path.Combine(Application.dataPath, folderName);
        Directory.CreateDirectory(Path.Combine(basePath, "Heightmaps"));
        Directory.CreateDirectory(Path.Combine(basePath, "Masks"));

        StartCoroutine(GenerateBatch());
    }

    private IEnumerator GenerateBatch()
    {
        Terrain terrain = GetComponent<Terrain>();
        TerrainData terrainData = terrain ? terrain.terrainData : null;

        if (terrainData != null)
        {
            terrainData.heightmapResolution = resolution + 1;
            terrainData.size = new Vector3(resolution, terrainHeight, resolution);
        }

        for (int sampleIndex = 0; sampleIndex < datasetSize; sampleIndex++)
        {
            float waterLevel = Mathf.Clamp01(baseWaterLevel + Random.Range(-0.02f, 0.03f));
            float riverAngle = Random.Range(0f, Mathf.PI * 2f);

            Vector2 macroOffset = RandomOffset();
            Vector2 rollingOffset = RandomOffset();
            Vector2 moistureOffset = RandomOffset();
            Vector2 ridgeOffset = RandomOffset();
            Vector2 riverWarpOffset = RandomOffset();
            Vector2 riverOffset = RandomOffset();
            Vector2 lakeOffset = RandomOffset();
            Vector2 forestOffset = RandomOffset();
            Vector2 hillOffset = RandomOffset();

            Texture2D maskTexture = new Texture2D(resolution, resolution, TextureFormat.RGB24, false);
            Texture2D heightTexture = new Texture2D(resolution, resolution, TextureFormat.RFloat, false);

            float[,] heights = new float[resolution, resolution];
            float[,] moistureMap = new float[resolution, resolution];
            float[,] rockPotentialMap = new float[resolution, resolution];
            float[,] waterStrengthMap = new float[resolution, resolution];
            float[,] forestClusterMap = new float[resolution, resolution];

            for (int x = 0; x < resolution; x++)
            {
                for (int y = 0; y < resolution; y++)
                {
                    float u = x / (float)(resolution - 1);
                    float v = y / (float)(resolution - 1);

                    Vector2 macroCoords = WarpCoordinates(u, v, macroScale * 0.7f, 0.09f, macroOffset);
                    float macro = LayeredNoise(macroCoords.x, macroCoords.y, macroScale, macroOffset);
                    float rolling = LayeredNoise(u, v, rollingScale, rollingOffset);
                    float moisture = LayeredNoise(u, v, moistureScale, moistureOffset);
                    float ridges = RidgedNoise(u, v, ridgeScale, ridgeOffset);
                    float hills = LayeredNoise(u, v, Mathf.Max(rollingScale * 0.55f, 2.2f), hillOffset);

                    Vector2 riverCoords = RotateCoordinates(u - 0.5f, v - 0.5f, riverAngle);
                    riverCoords += Vector2.one * 0.5f;
                    riverCoords = WarpCoordinates(riverCoords.x, riverCoords.y, 2.8f, 0.05f, riverWarpOffset);

                    float riverNoise = Mathf.PerlinNoise((riverCoords.x * 1.15f) + riverOffset.x, (riverCoords.y * 7.4f) + riverOffset.y);
                    float riverWidth = Mathf.Lerp(riverWidthRange.x, riverWidthRange.y, moisture);
                    float riverStrength = Mathf.Clamp01(1f - (Mathf.Abs(riverNoise - 0.5f) / riverWidth));
                    riverStrength *= Mathf.Clamp01((macro - 0.18f) / 0.45f);
                    riverStrength *= Mathf.Clamp01(1f - (ridges * 0.72f));

                    float lakeNoise = Mathf.PerlinNoise((u * 1.7f) + lakeOffset.x, (v * 1.7f) + lakeOffset.y);
                    float lakeStrength = 0f;
                    if (macro < 0.5f && moisture > 0.55f)
                    {
                        lakeStrength = Mathf.Clamp01((lakeNoise - 0.68f) / 0.2f);
                    }

                    float basePlain = 0.16f + (macro * 0.15f);
                    float rollingContribution = (rolling - 0.5f) * 0.06f;
                    float hillBlend = Mathf.SmoothStep(0.28f, 0.82f, (macro * 0.82f) + (hills * 0.35f));
                    float hillContribution = Mathf.Pow(Mathf.Clamp01(hills), 1.3f) * hillBlend * 0.11f;
                    float ridgeMask = Mathf.SmoothStep(0.58f, 0.9f, (macro * 0.7f) + (ridges * 0.3f));
                    float mountainContribution = Mathf.Pow(ridges, 2.25f) * ridgeMask * 0.12f;

                    float height = basePlain + rollingContribution + hillContribution + mountainContribution;
                    float plainsStrength = Mathf.Clamp01((0.62f - macro) / 0.34f) * Mathf.Clamp01(1f - (ridges * 0.9f));
                    float flattenedBase = basePlain + ((rolling - 0.5f) * 0.03f);
                    height = Mathf.Lerp(height, flattenedBase, plainsStrength * 0.48f);

                    float waterStrength = Mathf.Max(riverStrength, lakeStrength * 0.95f);
                    if (waterStrength > 0.001f)
                    {
                        float channelDepth = 0.02f + (riverStrength * 0.03f) + (lakeStrength * 0.045f);
                        float targetHeight = waterLevel - channelDepth;
                        height = Mathf.Lerp(height, targetHeight, waterStrength);
                    }

                    height = Mathf.Clamp01(height);

                    heights[y, x] = height;
                    moistureMap[y, x] = moisture;
                    rockPotentialMap[y, x] = Mathf.Clamp01((ridges * 0.32f) + (ridgeMask * 0.26f) + (mountainContribution * 1.35f));
                    waterStrengthMap[y, x] = waterStrength;
                    forestClusterMap[y, x] = Mathf.Clamp01(
                        (LayeredNoise(u, v, 2.35f, forestOffset) * 0.72f) +
                        (Mathf.PerlinNoise((u * 4.2f) + forestOffset.x + 41.7f, (v * 4.2f) + forestOffset.y + 12.3f) * 0.28f)
                    );
                }
            }

            float[,] unityHeights = new float[resolution, resolution];

            for (int x = 0; x < resolution; x++)
            {
                for (int y = 0; y < resolution; y++)
                {
                    float height = heights[y, x];
                    float slope = EstimateSlope(heights, x, y);
                    float moisture = moistureMap[y, x];
                    float rockPotential = rockPotentialMap[y, x];
                    float waterStrength = waterStrengthMap[y, x];
                    float forestCluster = forestClusterMap[y, x];
                    float forestSuitability = (moisture * 0.46f) +
                                              (forestCluster * 0.34f) +
                                              ((1f - rockPotential) * 0.12f) +
                                              (Mathf.Clamp01((0.22f - slope) / 0.22f) * 0.08f);

                    bool isWater = (waterStrength > 0.35f) || (height <= waterLevel + 0.004f);
                    bool isRock = !isWater &&
                                  (rockPotential > 0.7f) &&
                                  ((slope > 0.16f) || (height > waterLevel + 0.4f));
                    bool isForest = !isWater && !isRock &&
                                    (height > waterLevel + 0.02f) &&
                                    (height < waterLevel + 0.44f) &&
                                    (slope < 0.22f) &&
                                    (forestSuitability > 0.5f);

                    Color maskColor = Color.black;
                    if (isWater)
                    {
                        maskColor = Color.blue;
                    }
                    else if (isRock)
                    {
                        maskColor = Color.green;
                    }
                    else if (isForest)
                    {
                        maskColor = Color.red;
                    }

                    maskTexture.SetPixel(x, y, maskColor);
                    heightTexture.SetPixel(x, y, new Color(height, 0f, 0f));
                    unityHeights[y, x] = height;
                }
            }

            maskTexture.Apply();
            heightTexture.Apply();

            byte[] maskBytes = maskTexture.EncodeToPNG();
            File.WriteAllBytes(Path.Combine(basePath, $"Masks/mask_{sampleIndex:0000}.png"), maskBytes);

            byte[] heightBytes = heightTexture.EncodeToEXR(Texture2D.EXRFlags.CompressZIP);
            File.WriteAllBytes(Path.Combine(basePath, $"Heightmaps/height_{sampleIndex:0000}.exr"), heightBytes);

            if (terrainData != null)
            {
                terrainData.SetHeights(0, 0, unityHeights);
            }

            Destroy(maskTexture);
            Destroy(heightTexture);

            yield return null;
        }

        Debug.Log($"[DatasetGenerator] Finished generating {datasetSize} landscape pairs.");
    }

    private Vector2 RandomOffset()
    {
        return new Vector2(Random.Range(0f, 5000f), Random.Range(0f, 5000f));
    }

    private float LayeredNoise(float x, float y, float scale, Vector2 offset)
    {
        float baseNoise = Mathf.PerlinNoise((x * scale) + offset.x, (y * scale) + offset.y);
        float detailNoise = Mathf.PerlinNoise((x * scale * 2f) + offset.x + 37.1f, (y * scale * 2f) + offset.y + 91.7f);
        return Mathf.Clamp01((baseNoise * 0.68f) + (detailNoise * 0.32f));
    }

    private float RidgedNoise(float x, float y, float scale, Vector2 offset)
    {
        float baseNoise = Mathf.PerlinNoise((x * scale) + offset.x, (y * scale) + offset.y);
        float ridged = 1f - Mathf.Abs((baseNoise * 2f) - 1f);
        float detail = Mathf.PerlinNoise((x * scale * 2.1f) + offset.x + 11.3f, (y * scale * 2.1f) + offset.y + 63.5f);
        return Mathf.Clamp01((ridged * 0.75f) + (detail * 0.25f));
    }

    private Vector2 WarpCoordinates(float x, float y, float scale, float strength, Vector2 offset)
    {
        float warpX = (Mathf.PerlinNoise((x * scale) + offset.x, (y * scale) + offset.y) - 0.5f) * strength;
        float warpY = (Mathf.PerlinNoise((x * scale) + offset.x + 58.4f, (y * scale) + offset.y + 19.6f) - 0.5f) * strength;
        return new Vector2(x + warpX, y + warpY);
    }

    private Vector2 RotateCoordinates(float x, float y, float angle)
    {
        float cosine = Mathf.Cos(angle);
        float sine = Mathf.Sin(angle);
        return new Vector2((x * cosine) - (y * sine), (x * sine) + (y * cosine));
    }

    private float EstimateSlope(float[,] heights, int x, int y)
    {
        int left = Mathf.Max(x - 1, 0);
        int right = Mathf.Min(x + 1, resolution - 1);
        int down = Mathf.Max(y - 1, 0);
        int up = Mathf.Min(y + 1, resolution - 1);

        float dx = Mathf.Abs(heights[y, right] - heights[y, left]);
        float dy = Mathf.Abs(heights[up, x] - heights[down, x]);
        return Mathf.Clamp01((dx + dy) * 18f);
    }
}
