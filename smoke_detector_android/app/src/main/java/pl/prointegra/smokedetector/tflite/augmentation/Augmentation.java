package pl.prointegra.smokedetector.tflite.augmentation;

import android.graphics.Bitmap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import pl.prointegra.smokedetector.tflite.BBoxTransformations;
import pl.prointegra.smokedetector.tflite.Predictions;

public class Augmentation {
  private final List<Transform> transforms;
  private final List<Transform> reverseTransforms;

  public Augmentation(List<Transform> transforms, List<Transform> reverseTransforms) {
    this.transforms = transforms;
    this.reverseTransforms = reverseTransforms;
  }

  public List<Bitmap> generateTransformedImages(Bitmap source) {
    List<Bitmap> transformedBitmaps = new ArrayList<>();
    for (Transform transform : transforms) {
      transformedBitmaps.add(transform.transform(source));
    }
    return transformedBitmaps;
  }

  public Transform getReverseTransform(int i) {
    return reverseTransforms.get(i);
  }

  public Predictions chooseBestResults(Bitmap imageToPredict, Transform reverseTransform,
                                       Predictions bestPrediction,
                                       Predictions currentPrediction,
                                       boolean maximizePrecision) {

//    for (int imageIndex = 0; imageIndex < bestPrediction.boxes.size(); imageIndex++) {
    boolean newBestResults = false;
    if (maximizePrecision) {
      if (calculateScoresAverage(currentPrediction.probabilities) >
        calculateScoresAverage(bestPrediction.probabilities)) {
        newBestResults = true;
      }
    } else {
      // if current scores have more predictions than previous best regardless of confidence scores, take current as new best
      if (currentPrediction.boxes.size() > bestPrediction.boxes.size())
        newBestResults = true;
      else if (currentPrediction.boxes.size() == bestPrediction.boxes.size()) {
        // if current scores have the same number of predictions as previous best, check average confidence score
        if (calculateScoresAverage(currentPrediction.probabilities) >
          calculateScoresAverage(bestPrediction.probabilities)) {
          newBestResults = true;
        }
      }
    }

    if (newBestResults) {
      List<INDArray> boxes = BBoxTransformations.bboxTransformMultipleBoxesFloat(currentPrediction.boxes);
      INDArray boxes2 = Nd4j.create(boxes, new long[]{boxes.size(), 4});
      INDArray transformedBoxes = reverseTransform.transform(imageToPredict, boxes2).boxes;
      List<INDArray> finalBoxes = BBoxTransformations.bboxTransformMultipleBoxesToCXCYWH(transformedBoxes);
      return new Predictions(finalBoxes, currentPrediction.probabilities, currentPrediction.classIndices, true);
    }
    return bestPrediction;
  }
  private float calculateScoresAverage(List<Float> scores) {
    float sum = 0.0f;
    int count = 0;

    for (Float score : scores) {
      sum += score;
      count += 1;
    }

    if (count == 0) {
      return 0.0f;
    }
    float average = sum / count;
    return average;
  }

}