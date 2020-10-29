package pl.prointegra.smokedetector.tflite;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class BBoxTransformations {
  public static INDArray[] bboxTransform(INDArray[] bbox) {
    INDArray[] out = new INDArray[4];
    INDArray cx = bbox[0];
    INDArray cy = bbox[1];
    INDArray w = bbox[2];
    INDArray h = bbox[3];
    out[0] = cx.sub(w.div(2));
    out[1] = cy.sub(h.div(2));
    out[2] = cx.add(w.div(2));
    out[3] = cy.add(h.div(2));
    return out;
  }

  public static INDArray bboxTransformSingleBox(INDArray bbox) {
    INDArray out = Nd4j.createUninitialized(4);
    float cx = bbox.getFloat(0);
    float cy = bbox.getFloat(1);
    float w = bbox.getFloat(2);
    float h = bbox.getFloat(3);
    out.putScalar(0, (int) Math.floor(cx - (w / 2)));
    out.putScalar(1, (int) Math.floor(cy - (h / 2)));
    out.putScalar(2, (int) Math.floor(cx + (w / 2)));
    out.putScalar(3, (int) Math.floor(cy + (h / 2)));
    return out;
  }

  public static INDArray bboxTransformSingleBoxFloat(INDArray bbox) {
    INDArray out = Nd4j.createUninitialized(4);
    float cx = bbox.getFloat(0);
    float cy = bbox.getFloat(1);
    float w = bbox.getFloat(2);
    float h = bbox.getFloat(3);
    out.putScalar(0, (float) (cx - (w / 2)));
    out.putScalar(1, (float) (cy - (h / 2)));
    out.putScalar(2, (float) (cx + (w / 2)));
    out.putScalar(3, (float) (cy + (h / 2)));
    return out;
  }

  public static List<INDArray> bboxTransformMultipleBoxesFloat(List<INDArray> bboxes) {
    List<INDArray> transformedBBoxes = new ArrayList<>();
    for (INDArray bbox : bboxes) {
      transformedBBoxes.add(bboxTransformSingleBoxFloat(bbox));
    }
    return transformedBBoxes;
  }

  public static INDArray[] bboxTransformInverse(INDArray[] bbox) {
    INDArray[] out = new INDArray[4];
    INDArray xmin = bbox[0];
    INDArray ymin = bbox[1];
    INDArray xmax = bbox[2];
    INDArray ymax = bbox[3];
    INDArray width = xmax.sub(xmin).add(1.0);
    INDArray height = ymax.sub(ymin).add(1.0);
    out[0] = xmin.add(width.mul(0.5));
    out[1] = ymin.add(height.mul(0.5));
    out[2] = width;
    out[3] = height;
    return out;
  }

  /**
   * Converts a bbox of form [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
   */
  public static INDArray bboxTransformSingleBoxToCXCYWH(INDArray bbox) {
    INDArray out = Nd4j.createUninitialized(4);
    float xmin = bbox.getFloat(0);
    float ymin = bbox.getFloat(1);
    float xmax = bbox.getFloat(2);
    float ymax = bbox.getFloat(3);

    float width = xmax - xmin;
    float height = ymax - ymin;

    out.putScalar(0, (int) (Math.floor(xmin + (width / 2))));
    out.putScalar(1, (int) (Math.floor(ymin + (height / 2))));
    out.putScalar(2, (int) (Math.floor(width)));
    out.putScalar(3, (int) (Math.floor(height)));
    return out;
  }

  public static List<INDArray> bboxTransformMultipleBoxesToCXCYWH(List<INDArray> bboxes) {
    List<INDArray> transformedBBoxes = new ArrayList<>();
    for (INDArray bbox : bboxes) {
      transformedBBoxes.add(bboxTransformSingleBoxToCXCYWH(bbox));
    }
    return transformedBBoxes;
  }

  public static List<INDArray> bboxTransformMultipleBoxesToCXCYWH(INDArray bboxes) {
    List<INDArray> transformedBBoxes = new ArrayList<>();
    for (int i = 0; i < bboxes.size(0); i++) {
      transformedBBoxes.add(bboxTransformSingleBoxToCXCYWH(bboxes.get(point(i))));
    }
    return transformedBBoxes;
  }
}
