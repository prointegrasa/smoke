package pl.prointegra.smokedetector.tflite;

public class MathUtils {
  public static float mean(float[] m) {
    float sum = 0;
    for (float v : m) {
      sum += v;
    }
    return sum / m.length;
  }

  public static float getStd(float[] data, double mean) {
    double sum = 0;
    for (int index = 0; index != data.length; ++index) {
      sum += Math.pow(Math.abs(mean - data[index]), 2);
    }
    return (float) Math.sqrt(sum / data.length);
  }
}
