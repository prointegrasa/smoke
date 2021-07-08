// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate
import AVKit

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  var label: String? = nil
  let inferenceTime: Double
  let inferences: [Inference]
  let scaledBuffer: CVPixelBuffer
}

/// Stores one formatted inference.
struct Inference {
  let confidence: Float
  let className: String
  let rect: CGRect
  let displayColor: UIColor
}

struct RecLabel {
  let label: String
  let color: UIColor
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the SmokeDetection model.
enum SmokeDetection {
  static let modelInfo: FileInfo = (name: "model_float16", extension: "tflite")
}

/// Information about the FuelDetection model.
enum FuelDetection {
  static let modelInfo: FileInfo = (name: "fuel_classification_model", extension: "tflite")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {
  
  // MARK: - Internal Properties
  /// The current thread count used by the TensorFlow Lite Interpreter.
  public var ttaEnabled = true
  public var threshold: Float = 0.6
  public var postProcess: PostProcessor
  public let modelFile: FileInfo
  
  public var isSmokeDetectionModel: Bool {
    get {
      return modelFile.name == SmokeDetection.modelInfo.name
    }
  }
  
  let threadCount: Int
  let threadCountLimit = 10
  
  let inputWidth: Int!
  let inputHeight: Int!
  
  // MARK: Private properties
  private var labels: [String] = []
  
  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter
  
  private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
  private let rgbPixelChannels = 3
  private let dangerLevels = [
    0: RecLabel(label: "Danger: Very Low", color: .green),
    1: RecLabel(label: "Danger: Low", color: .green),
    2: RecLabel(label: "Danger: Medium", color: .orange),
    3: RecLabel(label: "Danger: High", color: .orange),
    4: RecLabel(label: "Danger: Very High", color: .red),
    5: RecLabel(label: "Danger: Extreme", color: .red),
  ]
  
  // MARK: - Initialization
  
  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, postProcessor: PostProcessor, threadCount: Int = 2, inputWidth: Int, inputHeight: Int) {
    self.modelFile = modelFileInfo
    self.inputWidth = inputWidth
    self.inputHeight = inputHeight
    self.postProcess = postProcessor
    
    let modelFilename = modelFileInfo.name
    
    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }
    
    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    var options = Interpreter.Options()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }
    super.init()
    
    // Load the classes listed in the labels file.
    // loadLabels(fileInfo: labelsFileInfo)
  }
  
  /// This class handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
  /// results for a successful inference.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
            sourcePixelFormat == kCVPixelFormatType_32BGRA ||
            sourcePixelFormat == kCVPixelFormatType_32RGBA)
    
    
    let predictionInvoker = PredictionInvoker(interpreter: interpreter, inputHeight: inputHeight, inputWidth: inputWidth)
    let imageChannels = 4
    assert(imageChannels >= predictionInvoker.inputChannels)
    
    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let videoRect = CGRect(x: 0, y: 0, width: imageWidth, height: imageHeight)
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    
    var croppingRect: CGRect? = nil
    if isSmokeDetectionModel {
      croppingRect = AVMakeRect(aspectRatio: scaledSize, insideRect: videoRect)
    }
    guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize, croppingRect: croppingRect) else {
      return nil
    }
    //let img = UIImage(ciImage: CIImage(cvPixelBuffer: scaledPixelBuffer))
    let startDate = Date()
    guard let modelOut = predictionInvoker.invoke(scaledPixelBuffer: scaledPixelBuffer) else {
      return nil
    }
    
    //TODO this needs refactor !!!
    if isSmokeDetectionModel {
      return self.getResultForSmokeDetection(modelOut, scaledPixelBuffer: scaledPixelBuffer, predictionInvoker: predictionInvoker, startDate: startDate, imageHeight: imageHeight, imageWidth: imageWidth)
    } else {
      guard let result = postProcess.invoke(modelOut: modelOut) as! String? else {
        return nil
      }
      
      let interval: TimeInterval = Date().timeIntervalSince(startDate) * 1000
      return Result(label: result, inferenceTime: interval, inferences: [], scaledBuffer: scaledPixelBuffer)
    }
  }
  
  private func getResultForSmokeDetection(_ modelOut: Tensor, scaledPixelBuffer: CVPixelBuffer, predictionInvoker: PredictionInvoker, startDate: Date, imageHeight: Int, imageWidth: Int) -> Result? {
    var inferences: [Inference] = []
    guard let prediction = postProcess.invoke(modelOut: modelOut) as! PredictionBoxes? else {
      return nil
    }
    var bestPrediction = prediction
    if ttaEnabled {
      let ttaService = TTAService(predictionInvoker: predictionInvoker, postProcessor: postProcess as! ModelPostProcessor)
      if let foundPrediction = ttaService.findBetterPrediction(forImage: scaledPixelBuffer, withPrediction: prediction) {
        bestPrediction = foundPrediction
      }
    }
    
    bestPrediction = rearrangeBoxes(onPrediction: bestPrediction)
    inferences = piResultToApplication(pred: bestPrediction, imageWidth: imageWidth, imageHeight: imageHeight)
    
    let interval: TimeInterval = Date().timeIntervalSince(startDate) * 1000
    // Returns the inference time and inferences
    return Result(inferenceTime: interval, inferences: inferences, scaledBuffer: scaledPixelBuffer)
  }
  
  func rearrangeBoxes(onPrediction: PredictionBoxes) -> PredictionBoxes {
    var pred = onPrediction
    pred.boxes = pred.boxes
      .map { box in
        return (postProcess as! ModelPostProcessor).mapBestPredictionBox(box: box)
      }
      .compactMap{ $0 }
    return pred
  }
  
  func piResultToApplication(pred: PredictionBoxes, imageWidth: Int, imageHeight: Int) -> [Inference] {
    let imageArea = 480 * 480
    var result: [Inference] = []
    
    for i in 0..<pred.boxes.count {
      let score = pred.probabilities[i]
      guard score >= threshold else {
        continue
      }
      
      let box = pred.boxes[i].toArray() as! [Int]
      let label = createRecognitionLabel(imageArea: imageArea, objectClass: pred.classIndexes[i], box: box)
      var rect = CGRect.zero
      rect.origin.x = CGFloat(box[0])
      rect.origin.y = CGFloat(box[1])
      rect.size.height = CGFloat(box[3]) - rect.origin.y
      rect.size.width = CGFloat(box[2]) - rect.origin.x
      
      var newRect = rect.applying(CGAffineTransform(scaleX: CGFloat(1.0 / 480.0), y: CGFloat(1.0 / 480.0)).concatenating(CGAffineTransform(scaleX: CGFloat(imageWidth), y: CGFloat(imageWidth))))
      newRect.origin.y +=  420 // 420 move from top to detection rect start
      result.append(Inference(
                      confidence: score,
                      className: label.label + (pred.isAugmented ? "_TTA" : ""),
                      rect: newRect,
                      displayColor: label.color)
      )
    }
    return result
  }
  
  private func createRecognitionLabel(imageArea: Int, objectClass: Int, box: [Int]) -> RecLabel {
    let objectArea = ( box[2] - box[0] ) * ( box[3] - box[1] )
    let areaRatio = Float(objectArea) / Float(imageArea)
    let dangerLevel = getDangerLevel(objectClass: objectClass, areaRatio: areaRatio)
    return dangerLevels[dangerLevel]!
  }
  
  private func getDangerLevel(objectClass: Int, areaRatio: Float) -> Int {
    var dangerLevel: Int!
    if objectClass == 0 {
      dangerLevel = 1
      if areaRatio < 0.015 {
        dangerLevel -= 1
      } else if areaRatio > 0.04 {
        dangerLevel += 1
      }
    } else {
      dangerLevel = 4
      if areaRatio < 0.035 {
        dangerLevel -= 1
      } else if areaRatio > 0.095 {
        dangerLevel += 1
      }
    }
    return dangerLevel
  }
  
  /// Filters out all the results with confidence score < threshold and returns the top N results
  /// sorted in descending order.
  //  func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat) -> [Inference]{
  //    var resultsArray: [Inference] = []
  //    if (outputCount == 0) {
  //      return resultsArray
  //    }
  //    for i in 0...outputCount - 1 {
  //
  //      let score = outputScores[i]
  //
  //      // Filters results with confidence < threshold.
  //      guard score >= threshold else {
  //        continue
  //      }
  //
  //      // Gets the output class names for detected classes from labels list.
  //      let outputClassIndex = Int(outputClasses[i])
  //      let outputClass = labels[outputClassIndex + 1]
  //
  //      var rect: CGRect = CGRect.zero
  //
  //      // Translates the detected bounding box to CGRect.
  //      rect.origin.y = CGFloat(boundingBox[4*i])
  //      rect.origin.x = CGFloat(boundingBox[4*i+1])
  //      rect.size.height = CGFloat(boundingBox[4*i+2]) - rect.origin.y
  //      rect.size.width = CGFloat(boundingBox[4*i+3]) - rect.origin.x
  //
  //      // The detected corners are for model dimensions. So we scale the rect with respect to the
  //      // actual image dimensions.
  //      let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
  //
  //      // Gets the color assigned for the class
  //      let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
  //      let inference = Inference(confidence: score,
  //                                className: outputClass,
  //                                rect: newRect,
  //                                displayColor: colorToAssign)
  //      resultsArray.append(inference)
  //    }
  //
  //    // Sort results in descending order of confidence.
  //    resultsArray.sort { (first, second) -> Bool in
  //      return first.confidence  > second.confidence
  //    }
  //
  //    return resultsArray
  //  }
}

// MARK: - Extensions

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
  
  func asFloats() -> [Float] {
    let bytes = Array<UInt8>(unsafeData: self)!
    var floats = [Float]()
    for i in 0..<self.count {
      floats.append(Float(bytes[i]))
    }
    return floats
  }
}

public extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
  
  
}
