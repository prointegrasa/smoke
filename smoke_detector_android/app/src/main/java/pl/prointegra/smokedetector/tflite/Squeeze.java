package pl.prointegra.smokedetector.tflite;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.tensorflow.lite.Interpreter;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import pl.prointegra.smokedetector.env.ImageUtils;
import pl.prointegra.smokedetector.tflite.augmentation.Augmentation;
import pl.prointegra.smokedetector.tflite.augmentation.HorizontalFlip;
import pl.prointegra.smokedetector.tflite.augmentation.Scale;
import pl.prointegra.smokedetector.tflite.augmentation.Sequence;
import pl.prointegra.smokedetector.tflite.augmentation.Transform;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.nd4j.linalg.ops.transforms.Transforms.exp;

class SlicedPredictions {
  public final INDArray classProbabilities;
  public final INDArray conf;
  public final INDArray boxDelta;

  SlicedPredictions(INDArray classProbabilities, INDArray conf, INDArray boxDelta) {
    this.classProbabilities = classProbabilities;
    this.conf = conf;
    this.boxDelta = boxDelta;
  }
}

public class Squeeze {
  private SqueezeConfig c;
  private Interpreter tfLite;
  private Augmentation augmentation;
  private int[] intValues;
  private float[] floatValues;
  private ByteBuffer imgData;
  private int inputSize;
  private final FrameConfiguration frameConfiguration;
  private Matrix frameToCropTransform;
  private long[] outputShape = new long[]{1, 14400, 7};
  private int outputSize = 1 * 14400 * 7;
  private RecognitionFactory recognitionFactory;

  public Squeeze(SqueezeConfig config, FrameConfiguration frameConfiguration, Interpreter tfLite) {
    this.c = config;
    this.tfLite = tfLite;
    this.frameConfiguration = frameConfiguration;
    this.augmentation = new Augmentation(generateTransforms(), generateReverseTransforms());
    this.recognitionFactory = new RecognitionFactory(config);
    frameToCropTransform = ImageUtils.getTransformationMatrix(
      frameConfiguration.width, frameConfiguration.height,
      c.IMAGE_WIDTH, c.IMAGE_HEIGHT,
      frameConfiguration.sensorOrientation, true);

    imgData = ByteBuffer.allocateDirect(c.IMAGE_WIDTH * c.IMAGE_HEIGHT * 3 * c.bytesPerChannel);
    imgData.order(ByteOrder.nativeOrder());
    intValues = new int[c.IMAGE_WIDTH * c.IMAGE_HEIGHT];
    floatValues = new float[c.IMAGE_WIDTH * c.IMAGE_HEIGHT * 3];
    inputSize = config.IMAGE_WIDTH; // TMP
  }

  public Buffer preprocessBitmap(Bitmap bitmap) {
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0,
      bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (c.isModelQuantized) { // Quantized model
//          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
//          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
//          imgData.put((byte) (pixelValue & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
        } else { // Float model
//          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          floatValues[(i * inputSize + j) * 3 + 0] = ((pixelValue & 0xFF));
          floatValues[(i * inputSize + j) * 3 + 1] = (((pixelValue >> 8) & 0xFF));
          floatValues[(i * inputSize + j) * 3 + 2] = (((pixelValue >> 16) & 0xFF));
        }
      }
    }

    long startTime = System.nanoTime();
    if (!c.isModelQuantized) {
      float imageMean = MathUtils.mean(floatValues);
      float imageStd = MathUtils.getStd(floatValues, imageMean);
      for (int i = 0; i < floatValues.length; ++i) {
        floatValues[i] = (floatValues[i] - imageMean) / imageStd;
      }
    }
    System.out.println("Calculating means/std took: " + (System.nanoTime() - startTime) / 1000000);

    return c.isModelQuantized ? imgData : FloatBuffer.wrap(floatValues);
  }

  private Bitmap resizeAndRotateBitmap(Bitmap bitmap) {
    Bitmap croppedBitmap = Bitmap.createBitmap(c.IMAGE_WIDTH, c.IMAGE_HEIGHT, Bitmap.Config.ARGB_8888);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(bitmap, frameToCropTransform, null);
    return croppedBitmap;
  }

  private Bitmap cropBitmap(Bitmap bitmap) {
    Bitmap croppedBitmap = Bitmap.createBitmap(c.IMAGE_WIDTH, c.IMAGE_HEIGHT, Bitmap.Config.ARGB_8888);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(bitmap, new Matrix(), null);
    return croppedBitmap;
  }

  public ArrayList<Recognition> predict(Bitmap bitmap) {
    long startTime;

    Bitmap transformedBitmap = resizeAndRotateBitmap(bitmap);
    Predictions finalPrediction = doPrediction(transformedBitmap);
    Predictions bestPrediction = finalPrediction;

    if (c.TTA_ENABLED) {
      startTime = System.nanoTime();
      List<Bitmap> transformedImages = augmentation.generateTransformedImages(transformedBitmap);
      for (int i = 0; i < transformedImages.size(); i++) {
        Bitmap transformedImage = transformedImages.get(i);
        transformedImage = cropBitmap(transformedImage);
        Transform reverseTransform = augmentation.getReverseTransform(i);
        Predictions prediction = doPrediction(transformedImage);
        bestPrediction = augmentation.chooseBestResults(transformedImage, reverseTransform,
          bestPrediction, prediction, c.TTA_MAXIMIZE_PRECISION);
      }
      System.out.println("Augmentation took: " + (System.nanoTime() - startTime) / 1000000);
    }

    for (int i = 0; i < bestPrediction.boxes.size(); i++) {
      bestPrediction.boxes.set(i, BBoxTransformations.bboxTransformSingleBox(bestPrediction.boxes.get(i)));
    }
    final ArrayList<Recognition> recognitions = recognitionFactory.createRecognitions(bestPrediction);
    return recognitions;
  }

  private Predictions doPrediction(Bitmap bitmap) {
    long startTime;
    Buffer input = preprocessBitmap(bitmap);

    float[] output = new float[outputSize];
    ByteBuffer out = ByteBuffer.allocateDirect(outputSize * c.bytesPerChannel);
    out.order(ByteOrder.nativeOrder());

    startTime = System.nanoTime();
    tfLite.run(input, out);
    System.out.println("Prediction took: " + (System.nanoTime() - startTime) / 1000000);
    out.rewind();
    out.asFloatBuffer().get(output);

    startTime = System.nanoTime();
    Predictions finalPredictions = postProcess(output, outputShape);
    System.out.println("Postprocessing took: " + (System.nanoTime() - startTime) / 1000000);
    return finalPredictions;
  }


  public Predictions postProcess(float[] output, long[] shape) {
    SlicedPredictions predictions = slicePredictions(output, shape);
    INDArray detBoxes = boxesFromDeltas(predictions.boxDelta);
    INDArray probs = predictions.classProbabilities.mul(predictions.conf.reshape(c.ANCHORS, 1)); // batch omitted
    INDArray detProbs = probs.max(1);
    INDArray detClass = probs.argMax(1);

    Predictions finalPredictions = filterPrediction(detBoxes, detProbs, detClass);
    return finalPredictions;
  }

  private SlicedPredictions slicePredictions(float[] predictions, long[] shape) {
    INDArray pred = Nd4j.create(predictions, shape);
    int nOutputs = c.CLASSES + 1 + 4;
    pred = pred.get(all(), all(), interval(0, nOutputs));
    pred = pred.reshape(c.N_ANCHORS_HEIGHT, c.N_ANCHORS_WIDTH, -1);
    int numClassProbs = c.ANCHOR_PER_GRID * c.CLASSES;

    INDArray predClassProbs = pred.get(all(), all(), interval(0, numClassProbs))
      .reshape(-1L, c.CLASSES);
    predClassProbs = Transforms.softmax(predClassProbs).reshape(c.ANCHORS, c.CLASSES);

    int numConfidenceScores = c.ANCHOR_PER_GRID + numClassProbs;

    INDArray predConf = pred.get(all(), all(), interval(numClassProbs, numConfidenceScores))
      .reshape(c.ANCHORS);
    predConf = Transforms.sigmoid(predConf);

    INDArray predBoxDelta = pred.get(all(), all(), interval(numConfidenceScores, pred.size(2))) // ???
      .reshape(c.ANCHORS, 4);
    return new SlicedPredictions(predClassProbs, predConf, predBoxDelta);
  }


  private INDArray safeExp(INDArray w, float threshold) {
    double slope = Math.exp(threshold);
    INDArray linBool = w.gt(threshold).castTo(DataType.FLOAT);
    INDArray linRegion = linBool.dup();
    INDArray linOut = w.sub(threshold - 1.f).mul(slope);
    INDArray expOut = exp(w.replaceWhere(Nd4j.zerosLike(w), Conditions.greaterThan(threshold)));
    INDArray out = linRegion.mul(linOut).add(Nd4j.onesLike(linRegion).sub(linRegion).mul(expOut));
    return out;
  }

  private INDArray boxesFromDeltas(INDArray predBoxDelta) {
    INDArray deltaX = predBoxDelta.get(all(), point(0));
    INDArray deltaY = predBoxDelta.get(all(), point(1));
    INDArray deltaW = predBoxDelta.get(all(), point(2));
    INDArray deltaH = predBoxDelta.get(all(), point(3));

    // get the coordinates and sizes of the anchor boxes from c
    INDArray anchorX = c.ANCHOR_BOX.get(all(), point(0));
    INDArray anchorY = c.ANCHOR_BOX.get(all(), point(1));
    INDArray anchorW = c.ANCHOR_BOX.get(all(), point(2));
    INDArray anchorH = c.ANCHOR_BOX.get(all(), point(3));

    // as we only predict the deltas, we need to transform the anchor box values before computing the loss
    INDArray boxCenterX = anchorX.add(deltaX.mul(anchorW));
    INDArray boxCenterY = anchorY.add(deltaY.mul(anchorH));
    INDArray boxWidth = anchorW.mul(safeExp(deltaW, c.EXP_THRESH));
    INDArray boxHeight = anchorH.mul(safeExp(deltaH, c.EXP_THRESH));

    INDArray[] transformedBboxes = BBoxTransformations.bboxTransform(new INDArray[]{boxCenterX, boxCenterY, boxWidth, boxHeight});
    INDArray xmins = transformedBboxes[0];
    INDArray ymins = transformedBboxes[1];
    INDArray xmaxs = transformedBboxes[2];
    INDArray ymaxs = transformedBboxes[3];

    xmins = Transforms.min(Transforms.max(xmins, 0.0), c.IMAGE_WIDTH - 1.0f);
    ymins = Transforms.min(Transforms.max(ymins, 0.0), c.IMAGE_HEIGHT - 1.0f);
    xmaxs = Transforms.max(Transforms.min(xmaxs, c.IMAGE_WIDTH - 1.0f), 0.0f);
    ymaxs = Transforms.max(Transforms.min(ymaxs, c.IMAGE_HEIGHT - 1.0f), 0.0f);

    INDArray detBoxes = Nd4j.vstack(
      BBoxTransformations.bboxTransformInverse(new INDArray[]{xmins, ymins, xmaxs, ymaxs}))
      .permute(1, 0); //changed to 2-dim
    return detBoxes;
  }

  private INDArray batchIou(INDArray boxes, INDArray box) {
    INDArray boxesCX = boxes.get(all(), point(0));
    INDArray boxesW = boxes.get(all(), point(2));
    INDArray boxesCY = boxes.get(all(), point(1));
    INDArray boxesH = boxes.get(all(), point(3));

    INDArray boxCX = box.get(point(0));
    INDArray boxW = box.get(point(2));
    INDArray boxCY = box.get(point(1));
    INDArray boxH = box.get(point(3));

    INDArray lr = Transforms.max(
      Transforms.min(
        boxesCX.add(boxesW.mul(0.5)),
        boxCX.add(boxW.mul(0.5))
      ).sub(
        Transforms.max(
          boxesCX.sub(boxesW.mul(0.5)),
          boxCX.sub(boxW.mul(0.5))
        )
      ), 0);

    INDArray tb = Transforms.max(
      Transforms.min(
        boxesCY.add(boxesH.mul(0.5)),
        boxCY.add(boxH.mul(0.5))
      ).sub(
        Transforms.max(
          boxesCY.sub(boxesH.mul(0.5)),
          boxCY.sub(boxH.mul(0.5))
        )
      ), 0);

    INDArray inter = lr.muli(tb);
    INDArray union = boxesW.mul(boxesH).addi(boxW.mul(boxH)).subi(inter);
    return inter.divi(union);
  }

  private INDArray nms(INDArray boxes, INDArray probabilities, float threshold) {
    INDArray order = Nd4j.sortWithIndices(probabilities.dup(), 0, false)[0];
    INDArray keep = Nd4j.ones(order.size(0));
    INDArray boxesSorted = Nd4jUtils.getForIndices(boxes, order.reshape(order.size(0)));

    for (int i = 0; i < order.size(0) - 1; ++i) {
      INDArray boxesSubset = boxesSorted.get(interval(i + 1, order.size(0)));
      INDArray ovps = batchIou(boxesSubset, boxes.get(point(i)));
      for (int j = 0; j < ovps.size(0); ++j) {
        if (ovps.getFloat(j) > threshold) {
          keep.putScalar(order.getInt(i + j + 1), 0.f);
        }
      }
    }
    return keep;
  }

  private INDArray nmsV2(INDArray boxes, INDArray probabilities, float threshold) {
    INDArray order = Nd4j.sortWithIndices(probabilities.dup(), 0, false)[0];
    INDArray keep = Nd4j.ones(order.size(0));

    for (int i = 0; i < order.size(0) - 1; ++i) {
      INDArray boxesIndices = order.get(interval(i + 1, order.size(0)));
      INDArray boxesSubset = Nd4jUtils.getForIndices(boxes, boxesIndices);
      INDArray ovps = batchIou(boxesSubset, boxes.get(point(i)));
      for (int j = 0; j < ovps.size(0); ++j) {
        if (ovps.getFloat(j) > threshold) {
          keep.putScalar(order.getInt(i + j + 1), 0.f);
        }
      }
    }
    return keep;
  }

  private Predictions filterPrediction(INDArray boxes, INDArray probabilities, INDArray classIndices) {
    INDArray indicesInOrder;
    if (c.TOP_N_DETECTION < probabilities.size(0) && c.TOP_N_DETECTION > 0) {
      indicesInOrder = Nd4j.sortWithIndices(probabilities.dup(), 0, false)[0]
        .get(interval(0, c.TOP_N_DETECTION));
    } else {
      // not checked
      indicesInOrder = Nd4j.where(probabilities.gt(c.PROB_THRESH), null, null)[0];
    }

    probabilities = probabilities.reshape(probabilities.size(0), 1).get(indicesInOrder)
      .reshape(indicesInOrder.size(0));
    boxes = boxes.get(indicesInOrder);
    classIndices = classIndices.reshape(classIndices.size(0), 1).get(indicesInOrder)
      .reshape(indicesInOrder.size(0));

    List<INDArray> finalBoxes = new ArrayList<>();
    List<Float> finalProbabilities = new ArrayList<>();
    List<Integer> finalClassIndices = new ArrayList<>();

    for (int class_ = 0; class_ < c.CLASSES; ++class_) {
      INDArray indicesPerClass = Nd4j.where(classIndices.eq(class_), null, null)[0];

      if (indicesPerClass.isEmpty()) {
        continue;
      }

      INDArray boxesSubset = Nd4jUtils.getForIndices(boxes, indicesPerClass);

      // do non maximum suppression
      long startTime = System.nanoTime();
      INDArray keep = nms(boxesSubset, Nd4jUtils.getForIndices(probabilities, indicesPerClass), c.NMS_THRESH);
      System.out.println("NMS took: " + (System.nanoTime() - startTime) / 1000000);
      for (int i = 0; i < keep.size(0); ++i) {
        if (Math.abs(keep.getFloat(i) - 1) < c.EPSILON) { // keep.getFloat(i) == 1
          float probability = probabilities.getFloat(indicesPerClass.getInt(i));
          if (probability > (float) c.FINAL_THRESHOLD) {
            finalBoxes.add(boxes.get(indicesPerClass.get(point(i)).reshape(1, 1)));
            finalProbabilities.add(probability);
            finalClassIndices.add(class_);
          }
        }
      }
    }

    return new Predictions(finalBoxes, finalProbabilities, finalClassIndices);
  }

  private List<Transform> generateTransforms() {
    List<Transform> ttaTransforms = Arrays.asList(
      new Sequence(Arrays.asList(new HorizontalFlip())),
      new Sequence(Arrays.asList(new Scale(-0.2f, -0.2f))),
      new Sequence(Arrays.asList(new HorizontalFlip(), new Scale(-0.2f, -0.2f))),
      new Sequence(Arrays.asList(new Scale(0.2f, 0.2f))),
      new Sequence(Arrays.asList(new HorizontalFlip(), new Scale(0.2f, 0.2f)))
    );
    return ttaTransforms;
  }

  private List<Transform> generateReverseTransforms() {
    List<Transform> ttaReverseTransforms = Arrays.asList(
      new Sequence(Arrays.asList(new HorizontalFlip())),
      new Sequence(Arrays.asList(new Scale(0.25f, 0.25f))),
      new Sequence(Arrays.asList(new Scale(0.25f, 0.25f), new HorizontalFlip())),
      new Sequence(Arrays.asList(new Scale(-0.1667f, -0.1667f))),
      new Sequence(Arrays.asList(new Scale(-0.1667f, -0.1667f), new HorizontalFlip()))
    );
    return ttaReverseTransforms;
  }

  public void setInterpreter(Interpreter interpreter) {
    this.tfLite = interpreter;
  }

  public void setTTAEnabled(boolean enabled) {
    this.c.TTA_ENABLED = enabled;
  }

  public void setShowDangerLevels(boolean showDangerLevels) {
    this.c.USE_PRED_FINAL_PRESENTATION = showDangerLevels;
  }

  public void setFinalThreshold(float finalThreshold) {
    this.c.FINAL_THRESHOLD = finalThreshold;
  }
}

