package pl.prointegra.smokedetector.tflite.augmentation;

import android.graphics.Bitmap;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Transform {
  Bitmap transform(Bitmap source);
  TransformationResult transform(Bitmap source, INDArray boxes);
}
