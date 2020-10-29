package pl.prointegra.smokedetector.tflite;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jUtils {
  public static INDArray getForIndices(INDArray array, INDArray indices) {
    if (indices.rank() == 0) {
      return Nd4j.createUninitialized(array.shape());
    }
    if (indices.isScalar()) {
      indices = indices.reshape(1, 1);
    }

    boolean isVectorOrScalar = array.isVectorOrScalar();
    boolean wasMatrix = array.isMatrix();
    if (isVectorOrScalar && array.rank() == 1) {
      array = array.reshape(array.size(0), 1);
    }

    array = array.get(indices);
    if (wasMatrix && array.isVectorOrScalar())
      array = array.reshape(1, array.size(0));
    else if (isVectorOrScalar && (array.rank() == 1 || array.rank() == 2 && array.size(1) == 1))
      array = array.reshape(array.size(0));
    return array;
  }
}
