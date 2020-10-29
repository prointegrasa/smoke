package pl.prointegra.smokedetector.tflite;

public class FrameConfiguration {
  final int width;
  final int height;
  final int sensorOrientation;

  public FrameConfiguration(int width, int height, int sensorOrientation) {
    this.width = width;
    this.height = height;
    this.sensorOrientation = sensorOrientation;
  }
}
