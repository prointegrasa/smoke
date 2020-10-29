package pl.prointegra.smokedetector.tflite;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class Predictions {
  public final List<INDArray> boxes;
  public final List<Float> probabilities;
  public final List<Integer> classIndices;
  public final boolean augmented;

  public Predictions(List<INDArray> boxes, List<Float> probabilities, List<Integer> classIndices, boolean augmented) {
    this.boxes = boxes;
    this.probabilities = probabilities;
    this.classIndices = classIndices;
    this.augmented = augmented;
  }

  public Predictions(List<INDArray> boxes, List<Float> probabilities, List<Integer> classIndices) {
    this(boxes, probabilities, classIndices, false);
  }
}
