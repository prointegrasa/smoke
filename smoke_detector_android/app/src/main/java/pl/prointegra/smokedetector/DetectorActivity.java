/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package pl.prointegra.smokedetector;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.widget.Toast;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

import pl.prointegra.smokedetector.customview.OverlayView;
import pl.prointegra.smokedetector.env.ImageUtils;
import pl.prointegra.smokedetector.env.Logger;
import pl.prointegra.smokedetector.tflite.Classifier;
import pl.prointegra.smokedetector.tflite.FrameConfiguration;
import pl.prointegra.smokedetector.tflite.Recognition;
import pl.prointegra.smokedetector.tflite.SqueezeClassifier;
import pl.prointegra.smokedetector.tflite.augmentation.BitmapTransformations;
import pl.prointegra.smokedetector.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final String MODEL_FILE = "model_float16.tflite";
  private static final boolean MAINTAIN_ASPECT = true;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(9999, 9999);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private String imagesFolderName;
  private int detectionNum = 0;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    tracker = new MultiBoxTracker(this);

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    try {
      detector = SqueezeClassifier.create(getAssets(),
        MODEL_FILE,
        config,
        new FrameConfiguration(previewWidth, previewHeight, sensorOrientation));
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
        Toast.makeText(
          getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, Config.ARGB_8888);

    frameToCropTransform =
      ImageUtils.getTransformationMatrix(
        previewWidth, previewHeight,
        config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
        sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(canvas -> {
      tracker.draw(canvas);
      if (isDebug()) {
        tracker.drawDebug(canvas);
      }
    });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    imagesFolderName = new SimpleDateFormat("dd-MM-yyyy HH:mm", Locale.US).format(new Date());
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmapToImages(croppedBitmap, "preview.jpeg");
    }

    runInBackground(() -> {
      LOGGER.i("Running detection on image " + currTimestamp);
      final long startTime = SystemClock.uptimeMillis();
      final List<Recognition> results = detector.recognizeImage(rgbFrameBitmap);
      lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

      cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
      final Canvas canvas1 = new Canvas(cropCopyBitmap);


      final List<Recognition> mappedRecognitions = new LinkedList<>();
      for (final Recognition result : results) {
        final RectF location = result.getLocation();
        if (location != null) {
          drawRecognitions(canvas1, result);

          cropToFrameTransform.mapRect(location);
          result.setLocation(location);
          mappedRecognitions.add(result);
        }
      }

      if (results.size() > 0) {
        tracker.setPreviewImage(Bitmap.createBitmap(rgbFrameBitmap));
      }

      tracker.trackResults(mappedRecognitions, currTimestamp);
      trackingOverlay.postInvalidate();

      if (results.size() > 0) {
        saveOriginalAndPredictedImage();
      }

      computingDetection = false;

      runOnUiThread(() -> {
        showFrameInfo(previewWidth + "x" + previewHeight);
        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
        showInference(lastProcessingTimeMs + "ms");
      });
    });
  }

  private void saveOriginalAndPredictedImage() {
    ++detectionNum;
    Bitmap rgbFrameRotatedBitmap = BitmapTransformations.rotate(rgbFrameBitmap, sensorOrientation);
    ImageUtils.saveBitmapToImages(rgbFrameRotatedBitmap, detectionNum + "_original.jpeg", imagesFolderName);
    ImageUtils.saveBitmapToImages(cropCopyBitmap, detectionNum + "_prediction.jpeg", imagesFolderName);
//    ImageUtils.saveBitmapToImages(croppedBitmap, detectionNum + "_cropped.jpeg", imagesFolderName);
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }

  @Override
  protected int getNumThreads() {
    return SqueezeClassifier.NUM_THREADS;
  }

  @Override
  protected void setUseGPU(final boolean useGPU) {
    runInBackground(() -> detector.setUseGPU(useGPU));
  }

  @Override
  protected void setUseTTA(final boolean useTTA) {
    runInBackground(() -> detector.setUseTTA(useTTA));
  }

  @Override
  protected void setShowDangerLevels(final boolean showDangerLevels) {
    runInBackground(() -> detector.setShowDangerLevels(showDangerLevels));
  }

  @Override
  protected void setFinalThreshold(final float finalThreshold) {
    runInBackground(() -> detector.setFinalThreshold(finalThreshold));
  }
}

