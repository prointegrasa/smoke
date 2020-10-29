/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package pl.prointegra.smokedetector.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Vector;

import pl.prointegra.smokedetector.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class SqueezeClassifier implements Classifier {
  private static final Logger LOGGER = new Logger();

  // Float model
  private static float IMAGE_MEAN = 128.0f;
  private static float IMAGE_STD = 128.0f;
  // Number of threads in the java app
  public static int NUM_THREADS = 6;
  private Vector<String> labels = new Vector<>();

  private Interpreter tfLite;

  private Squeeze squeeze;
  private MappedByteBuffer modelFile;

  private SqueezeClassifier() {
  }

  /**
   * Memory-map the model file in Assets.
   */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
          throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager  The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   */
  public static Classifier create(
          final AssetManager assetManager,
          final String modelFilename,
          final SqueezeConfig config,
          final FrameConfiguration frameConfiguration)
          throws IOException {
    final SqueezeClassifier d = new SqueezeClassifier();

    d.modelFile = loadModelFile(assetManager, modelFilename);
    d.tfLite = createInterpreter(d.modelFile, null);
    d.tfLite.setNumThreads(NUM_THREADS);
    d.squeeze = new Squeeze(config, frameConfiguration, d.tfLite);
    d.labels.addAll(config.CLASS_NAMES);

    return d;
  }

  private static Interpreter createInterpreter(MappedByteBuffer modelFile, Interpreter.Options options) {
    try {
       return new Interpreter(modelFile, options);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    return squeeze.predict(bitmap);
  }

  public void setNumThreads(int num_threads) {
    NUM_THREADS = num_threads;
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }

  @Override
  public void setUseGPU(boolean enabled) {
    if (tfLite != null) {
      tfLite.close();
      if (enabled)
        tfLite = createInterpreter(this.modelFile, (new Interpreter.Options()).addDelegate(new GpuDelegate()));
      else {
        tfLite = createInterpreter(this.modelFile, null);
        tfLite.setNumThreads(NUM_THREADS);
      }
      squeeze.setInterpreter(tfLite);
    }
  }

  @Override
  public void setUseTTA(boolean enabled) {
    squeeze.setTTAEnabled(enabled);
  }

  @Override
  public void setShowDangerLevels(boolean showDangerLevels) {
    squeeze.setShowDangerLevels(showDangerLevels);
  }

  @Override
  public void setFinalThreshold(float finalThreshold) {
    squeeze.setFinalThreshold(finalThreshold);
  }
}

