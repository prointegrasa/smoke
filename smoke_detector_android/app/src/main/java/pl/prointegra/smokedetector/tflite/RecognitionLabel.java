package pl.prointegra.smokedetector.tflite;

public class RecognitionLabel {
  private String text;
  private int color;

  public RecognitionLabel(String text, int color) {
    this.text = text;
    this.color = color;
  }

  public String getText() {
    return text;
  }

  public int getColor() {
    return color;
  }
}
