package pl.prointegra.smokedetector.tflite;

import android.graphics.Color;
import android.graphics.RectF;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class RecognitionFactory {
  private SqueezeConfig c;
  private Map<Integer, RecognitionLabel> dangerLevels = new HashMap<>();

  public RecognitionFactory(SqueezeConfig config) {
    this.c = config;
    this.buildDangerLevelsMap();
  }

  private void buildDangerLevelsMap() {
    dangerLevels.put(0, new RecognitionLabel("Danger: Very Low", Color.GREEN));
    dangerLevels.put(1, new RecognitionLabel("Danger: Low", Color.GREEN));
    dangerLevels.put(2, new RecognitionLabel("Danger: Medium", 0xFFFF8000));
    dangerLevels.put(3, new RecognitionLabel("Danger: High", 0xFFFF8000));
    dangerLevels.put(4, new RecognitionLabel("Danger: Very High", Color.RED));
    dangerLevels.put(5, new RecognitionLabel("Danger: Extreme", Color.RED));
  }

  public ArrayList<Recognition> createRecognitions(Predictions finalPredictions) {
    final ArrayList<Recognition> recognitions = new ArrayList<>();
    float imageArea = c.IMAGE_WIDTH * c.IMAGE_HEIGHT;
    for (int i = 0; i < finalPredictions.boxes.size(); ++i) {
      INDArray box = finalPredictions.boxes.get(i);
      RecognitionLabel label = createRecognitionLabel(finalPredictions, i, imageArea);
      float probability = finalPredictions.probabilities.get(i);
      float left = box.getFloat(0);
      float top = box.getFloat(1);
      float right = box.getFloat(2);
      float bottom = box.getFloat(3);
      System.out.println(label.getText() + " " + probability + " " + top + " " + left + " " + bottom + " " + right);
      recognitions.add(new Recognition("" + i, label, probability, new RectF(left, top, right, bottom)));
    }
    return recognitions;
  }

  private RecognitionLabel createRecognitionLabel(Predictions finalPredictions, int i, float imageArea) {
    Integer objectClass = finalPredictions.classIndices.get(i);
    if (c.USE_PRED_FINAL_PRESENTATION) {
      INDArray box = finalPredictions.boxes.get(i);
      float xmin = box.getFloat(0);
      float ymin = box.getFloat(1);
      float xmax = box.getFloat(2);
      float ymax = box.getFloat(3);
      float objectArea = (xmax - xmin) * (ymax - ymin);
      return createFinalPresentation(imageArea, objectArea, objectClass);
    } else {
      double probability = finalPredictions.probabilities.get(i);
      probability = Math.round(probability * 100.0) / 100.0;
      String label = c.CLASS_NAMES.get(objectClass) + probability;
      if (finalPredictions.augmented) {
        label += "TTA";
      }
      return new RecognitionLabel(label, Color.RED);
    }
  }

  private RecognitionLabel createFinalPresentation(float imageArea, float objectArea, int objectClass) {
    float areaRatio = objectArea / imageArea;
    int dangerLevel = getDangerLevel(objectClass, areaRatio);
    return dangerLevels.get(dangerLevel);
  }

  private int getDangerLevel(int objectClass, float areaRatio) {
    int dangerLevel;
    if (objectClass == 0) {
      dangerLevel = 1;
      if (areaRatio < 0.015) {
        dangerLevel -= 1;
      } else if (areaRatio > 0.04) {
        dangerLevel += 1;
      }
    } else {
      dangerLevel = 4;
      if (areaRatio < 0.035) {
        dangerLevel -= 1;
      } else if (areaRatio > 0.095) {
        dangerLevel += 1;
      }
    }
    return dangerLevel;
  }
}
