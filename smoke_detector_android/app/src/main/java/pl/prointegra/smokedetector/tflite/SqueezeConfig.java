package pl.prointegra.smokedetector.tflite;

import android.content.res.AssetManager;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class SqueezeConfig {
  public int CLASSES;
  public int ANCHORS_HEIGHT;
  public int N_ANCHORS_HEIGHT = ANCHORS_HEIGHT;
  public int ANCHORS_WIDTH;
  public int N_ANCHORS_WIDTH = ANCHORS_WIDTH;
  public int ANCHOR_PER_GRID;
  public int ANCHORS;
  public int IMAGE_WIDTH;
  public int IMAGE_HEIGHT;
  public float EXP_THRESH;
  public float PROB_THRESH;
  public float EPSILON;
  public float NMS_THRESH;
  public float FINAL_THRESHOLD;
  public long TOP_N_DETECTION;
  public List<String> CLASS_NAMES = new ArrayList<>();
  public INDArray ANCHOR_BOX;
  public float[] ANCHOR_SEED;
  public boolean isModelQuantized = false;
  public int bytesPerChannel;
  public boolean TTA_ENABLED = true;
  public boolean TTA_MAXIMIZE_PRECISION = true;
  public boolean USE_PRED_FINAL_PRESENTATION = true;

  private SqueezeConfig() {
    bytesPerChannel = isModelQuantized ? 1 : 4; // 4 for floating point
  }

  public static SqueezeConfig fromFile(AssetManager assetManager, String path) throws IOException {
    SqueezeConfig config = new SqueezeConfig();
    String configFileContents = loadJSONFromAsset(assetManager, path);
    try {
      JSONObject root = new JSONObject(configFileContents);
      config.ANCHORS_HEIGHT = root.getInt("ANCHORS_HEIGHT");
      config.N_ANCHORS_HEIGHT = config.ANCHORS_HEIGHT;
      config.ANCHORS_WIDTH = root.getInt("ANCHORS_WIDTH");
      config.N_ANCHORS_WIDTH = config.ANCHORS_WIDTH;
      config.ANCHOR_PER_GRID = root.getInt("ANCHOR_PER_GRID");
      loadAnchorSeed(config, root);
      config.CLASSES = root.getInt("CLASSES");
      loadClassNames(config, root);
      config.EPSILON = (float) root.getDouble("EPSILON");
      config.EXP_THRESH = (float) root.getDouble("EXP_THRESH");
      config.FINAL_THRESHOLD = (float) root.getDouble("FINAL_THRESHOLD");
      config.IMAGE_WIDTH = root.getInt("IMAGE_WIDTH_PROCESSING");
      config.IMAGE_HEIGHT = root.getInt("IMAGE_HEIGHT_PROCESSING");
      config.PROB_THRESH = (float) root.getDouble("PROB_THRESH");
      config.NMS_THRESH = (float) root.getDouble("NMS_THRESH");
      config.TOP_N_DETECTION = root.getInt("TOP_N_DETECTION");
      config.TTA_ENABLED = root.getInt("USE_TTA_ON_PREDICT") != 0;
      config.TTA_MAXIMIZE_PRECISION = root.getInt("USE_TTA_MAXIMIZE_PRECISION") != 0;
      config.USE_PRED_FINAL_PRESENTATION = root.getInt("USE_PRED_FINAL_PRESENTATION") != 0;

      long startTime = System.nanoTime();
      config.generateAnchorBoxFromSeed();
      System.out.println("Anchor box preparation took: " + (System.nanoTime() - startTime) / 1000000);

    } catch (JSONException e) {
      e.printStackTrace();
    }
    return config;
  }

  private static void loadClassNames(SqueezeConfig config, JSONObject root) throws JSONException {
    JSONArray classNames = root.getJSONArray("CLASS_NAMES");
    for (int i = 0; i < classNames.length(); i++) {
      config.CLASS_NAMES.add(classNames.getString(i));
    }
  }

  private static void loadAnchorSeed(SqueezeConfig config, JSONObject root) throws JSONException {
    JSONArray anchorSeed = root.getJSONArray("ANCHOR_SEED");
    float[] flatAnchorSeed = new float[anchorSeed.length() * 2];
    for (int i = 0; i < anchorSeed.length(); ++i) {
      JSONArray nested = anchorSeed.getJSONArray(i);
      for (int j = 0; j < nested.length(); ++j) {
        flatAnchorSeed[i * nested.length() + j] = (float) nested.getDouble(j);
      }
    }

    config.ANCHOR_SEED = flatAnchorSeed;
  }

  private void generateAnchorBoxFromSeed() {
    if (ANCHOR_BOX != null)
      return;

    INDArray seed = Nd4j.create(this.ANCHOR_SEED);
    int H = ANCHORS_HEIGHT, W = ANCHORS_WIDTH, B = ANCHOR_PER_GRID;
    INDArray anchorShapes = Nd4j.repeat(seed, W * H)
      .reshape(H, W, B, 2);
    INDArray centerX = Nd4j.repeat(
      Nd4j.linspace(1, W, W).muli((float) IMAGE_WIDTH / (W + 1)), H * B
    ).reshape(B, H, W).permute(1, 2, 0).reshape(H, W, B, 1);

    INDArray centerY = Nd4j.repeat(
      Nd4j.linspace(1, H, H).muli((float) IMAGE_HEIGHT / (H + 1)), W * B
    ).reshape(B, H, W).permute(2, 1, 0).reshape(H, W, B, 1);

    INDArray anchors = Nd4j.concat(3, centerX, centerY, anchorShapes)
      .reshape(-1, 4);

    ANCHOR_BOX = anchors;
    ANCHORS = (int) anchors.size(0);
  }

  private static String loadJSONFromAsset(AssetManager assetManager, String filePath) throws IOException {
    InputStream is = assetManager.open(filePath);
    int size = is.available();
    byte[] buffer = new byte[size];
    is.read(buffer);
    is.close();
    return new String(buffer, StandardCharsets.UTF_8);
  }
}
