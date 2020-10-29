package pl.prointegra.smokedetector.tflite.augmentation;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;

public class BitmapTransformations {

  public static Bitmap flipHorizontally(Bitmap source) {
    return scale(source, -1, 1);
  }

  public static Bitmap scale(Bitmap source, float sx, float sy) {
    Matrix matrix = new Matrix();
    matrix.postScale(sx, sy, source.getWidth() / 2f, source.getHeight() / 2f);
    return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
  }

  public static Bitmap scaleAndCrop(Bitmap source, float sx, float sy, int cropWidth, int cropHeight) {
    Matrix matrix = new Matrix();
    matrix.postScale(sx, sy, 0, 0);
    Bitmap croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(source, matrix, null);
    return croppedBitmap;
  }

  public static Bitmap rotate(Bitmap source, float degrees) {
    Matrix matrix = new Matrix();
    matrix.postRotate(degrees, source.getWidth() / 2f, source.getHeight() / 2f);
    return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
  }
}
