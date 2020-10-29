package pl.prointegra.smokedetector.tflite;

import android.graphics.RectF;

/** An immutable result returned by a Classifier describing what was recognized. */
public class Recognition {
  /**
   * A unique identifier for what has been recognized. Specific to the class, not the instance of
   * the object.
   */
  private final String id;

  /** Display name for the recognition. */
  private final RecognitionLabel label;

  /**
   * A sortable score for how good the recognition is relative to others. Higher should be better.
   */
  private final Float confidence;

  /** Optional location within the source image for the location of the recognized object. */
  private RectF location;

  public Recognition(
    final String id, final RecognitionLabel label, final Float confidence, final RectF location) {
    this.id = id;
    this.label = label;
    this.confidence = confidence;
    this.location = location;
  }

  public String getId() {
    return id;
  }

  public RecognitionLabel getLabel() {
    return label;
  }

  public Float getConfidence() {
    return confidence;
  }

  public RectF getLocation() {
    return new RectF(location);
  }

  public void setLocation(RectF location) {
    this.location = location;
  }

  @Override
  public String toString() {
    String resultString = "";
    if (id != null) {
      resultString += "[" + id + "] ";
    }

    if (label != null) {
      resultString += label.getText() + " ";
    }

    if (confidence != null) {
      resultString += String.format("(%.1f%%) ", confidence * 100.0f);
    }

    if (location != null) {
      resultString += location + " ";
    }

    return resultString.trim();
  }
}
