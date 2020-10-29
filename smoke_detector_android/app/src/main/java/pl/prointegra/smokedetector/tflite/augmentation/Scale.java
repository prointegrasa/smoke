package pl.prointegra.smokedetector.tflite.augmentation;

import android.graphics.Bitmap;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import pl.prointegra.smokedetector.tflite.Nd4jUtils;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class Scale implements Transform {
  private float scaleX;
  private float scaleY;

  public Scale(float scaleDeltaX, float scaleDeltaY) {
    this.scaleX = 1 + scaleDeltaX;
    this.scaleY = 1 + scaleDeltaY;
  }

  @Override
  public Bitmap transform(Bitmap source) {
    return BitmapTransformations.scaleAndCrop(source, scaleX, scaleY, source.getWidth(), source.getHeight());
  }

  @Override
  public TransformationResult transform(Bitmap source, INDArray boxes) {
    Bitmap transformedBitmap = transform(source);

    if (boxes != null && boxes.size(0) > 0) {
      INDArray transformedBoxes = boxes.dup();
      transformedBoxes.get(all(), interval(0, 4)).muli(Nd4j.create(new float[]{scaleX, scaleY, scaleX, scaleY}));
      transformedBoxes = clipBox(transformedBoxes, new float[]{0, 0, 1 + source.getWidth(), source.getHeight()}, 0.95f);
      return new TransformationResult(transformedBitmap, transformedBoxes);
    }
    return new TransformationResult(transformedBitmap, boxes);
  }

  /**
   * Clips the bounding boxes to the borders of an image
   */
  private INDArray clipBox(INDArray bboxes, float[] clipBox, float alpha) {
    INDArray bboxArea = bboxArea(bboxes);
    INDArray xMin = Transforms.max(bboxes.get(all(), point(0)), clipBox[0]).reshape(-1, 1);
    INDArray yMin = Transforms.max(bboxes.get(all(), point(1)), clipBox[1]).reshape(-1, 1);
    INDArray xMax = Transforms.min(bboxes.get(all(), point(2)), clipBox[2]).reshape(-1, 1);
    INDArray yMax = Transforms.min(bboxes.get(all(), point(3)), clipBox[3]).reshape(-1, 1);

    bboxes = Nd4j.hstack(xMin, yMin, xMax, yMax, bboxes.get(all(), interval(4, bboxes.size(1))));
    INDArray deltaArea = (bboxArea.sub(bboxArea(bboxes))).div(bboxArea);
    INDArray mask = deltaArea.lt(1 - alpha).castTo(DataType.INT);
    return Nd4jUtils.getForIndices(bboxes, Nd4j.where(mask.eq(1), null, null)[0]);
  }

  private INDArray bboxArea(INDArray bboxes) {
    return bboxes.get(all(), point(2)).sub(bboxes.get(all(), point(0)))
      .mul(bboxes.get(all(), point(3)).sub(bboxes.get(all(), point(1))));
  }
}
