package pl.prointegra.smokedetector.tflite.augmentation;

import android.graphics.Bitmap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.indices;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class HorizontalFlip implements Transform {
  @Override
  public Bitmap transform(Bitmap source) {
    return BitmapTransformations.flipHorizontally(source);
  }

  @Override
  public TransformationResult transform(Bitmap source, INDArray boxes) {
    Bitmap transformedBitmap = transform(source);
    INDArray imgCenter = Nd4j.create(new float[]{source.getWidth() / 2, source.getHeight() / 2}, 2);
    imgCenter = Nd4j.hstack(imgCenter, imgCenter);
    if (boxes == null || boxes.size(0) == 0)
      return new TransformationResult(transformedBitmap, boxes);

    INDArray transformedBoxes = boxes.dup();
    INDArray transformedIndices = transformedBoxes.get(all(), indices(0, 2))
      .addi(imgCenter.get(indices(0, 2)).sub(transformedBoxes.get(all(), indices(0, 2))).muli(2));
    transformedBoxes.putScalar(0, transformedIndices.getFloat(0));
    transformedBoxes.putScalar(2, transformedIndices.getFloat(1));
    INDArray boxW = Transforms.abs(transformedBoxes.get(all(), point(0)).sub(transformedBoxes.get(all(), point(2))));
    transformedBoxes.get(all(), point(0)).subi(boxW);
    transformedBoxes.get(all(), point(2)).addi(boxW);
    return new TransformationResult(transformedBitmap, transformedBoxes);
  }
}
