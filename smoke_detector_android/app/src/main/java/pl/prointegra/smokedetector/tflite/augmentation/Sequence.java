package pl.prointegra.smokedetector.tflite.augmentation;

import android.graphics.Bitmap;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class Sequence implements Transform {
  private List<Transform> transforms;

  public Sequence(List<Transform> transforms) {
    this.transforms = transforms;
  }

  @Override
  public Bitmap transform(Bitmap source) {
    Bitmap transformedBitmap = source;
    for (Transform transform : transforms) {
      transformedBitmap = transform.transform(transformedBitmap);
    }
    return transformedBitmap;
  }

  @Override
  public TransformationResult transform(Bitmap source, INDArray boxes) {
    TransformationResult transformationResult = new TransformationResult(source, boxes);
    for (Transform transform : transforms) {
      transformationResult = transform.transform(transformationResult.bitmap, transformationResult.boxes);
    }
    return transformationResult;

  }
}
