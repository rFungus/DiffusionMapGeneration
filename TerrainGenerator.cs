using UnityEngine;

public class TerrainGenerator : MonoBehaviour
{
    [Header("Terrain Size")]
    public int width = 256;
    public int length = 256;
    public int depth = 32;

    [Header("Shape")]
    public float macroScale = 1.5f;
    public float rollingScale = 5f;
    public float ridgeScale = 4.5f;
    [Range(0f, 1f)] public float plainsBias = 0.55f;

    private Vector2 macroOffset;
    private Vector2 rollingOffset;
    private Vector2 ridgeOffset;

    private void Start()
    {
        macroOffset = RandomOffset();
        rollingOffset = RandomOffset();
        ridgeOffset = RandomOffset();

        Terrain terrain = GetComponent<Terrain>();
        if (terrain == null || terrain.terrainData == null)
        {
            Debug.LogWarning("[TerrainGenerator] No Terrain component was found on this object. Generation was skipped.");
            enabled = false;
            return;
        }

        terrain.terrainData = GenerateTerrain(terrain.terrainData);
    }

    private TerrainData GenerateTerrain(TerrainData terrainData)
    {
        terrainData.heightmapResolution = width + 1;
        terrainData.size = new Vector3(width, depth, length);
        terrainData.SetHeights(0, 0, GenerateHeights());
        return terrainData;
    }

    private float[,] GenerateHeights()
    {
        float[,] heights = new float[length, width];

        for (int y = 0; y < length; y++)
        {
            for (int x = 0; x < width; x++)
            {
                heights[y, x] = CalculateHeight(x, y);
            }
        }

        return heights;
    }

    private float CalculateHeight(int x, int y)
    {
        float u = x / (float)(width - 1);
        float v = y / (float)(length - 1);

        float macro = LayeredNoise(u, v, macroScale, macroOffset);
        float rolling = LayeredNoise(u, v, rollingScale, rollingOffset);
        float ridges = RidgedNoise(u, v, ridgeScale, ridgeOffset);

        float plains = 0.18f + (macro * 0.18f);
        float rollingContribution = (rolling - 0.5f) * 0.09f;
        float mountainBlend = Mathf.SmoothStep(0.35f, 0.88f, (macro * 0.62f) + (ridges * 0.7f));
        float mountainContribution = Mathf.Pow(ridges, 1.4f) * mountainBlend * 0.24f;

        float height = plains + rollingContribution + mountainContribution;
        float flattened = Mathf.Lerp(height, 0.16f + (macro * 0.12f), plainsBias * Mathf.Clamp01(1f - ridges));

        return Mathf.Clamp01(flattened);
    }

    private float LayeredNoise(float x, float y, float scale, Vector2 offset)
    {
        float a = Mathf.PerlinNoise((x * scale) + offset.x, (y * scale) + offset.y);
        float b = Mathf.PerlinNoise((x * scale * 2f) + offset.x + 37.3f, (y * scale * 2f) + offset.y + 19.5f);
        return Mathf.Clamp01((a * 0.68f) + (b * 0.32f));
    }

    private float RidgedNoise(float x, float y, float scale, Vector2 offset)
    {
        float value = Mathf.PerlinNoise((x * scale) + offset.x, (y * scale) + offset.y);
        float ridge = 1f - Mathf.Abs((value * 2f) - 1f);
        return Mathf.Clamp01(ridge);
    }

    private Vector2 RandomOffset()
    {
        return new Vector2(Random.Range(0f, 5000f), Random.Range(0f, 5000f));
    }
}
