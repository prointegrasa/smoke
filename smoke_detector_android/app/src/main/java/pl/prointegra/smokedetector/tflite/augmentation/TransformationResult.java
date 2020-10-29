package pl.prointegra.smokedetector.tflite.augmentation;

import android.graphics.Bitmap;

import org.nd4j.linalg.api.ndarray.INDArray;

public class TransformationResult {
  public final Bitmap bitmap;
  public final INDArray boxes;

  public TransformationResult(Bitmap bitmap, INDArray boxes) {
    this.bitmap = bitmap;
    this.boxes = boxes;
  }
}
