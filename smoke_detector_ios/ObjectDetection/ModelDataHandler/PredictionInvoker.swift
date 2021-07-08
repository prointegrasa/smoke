//
//  PredictionInvoker.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 04/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import Foundation
import TensorFlowLite
import Accelerate
import CoreImage

class PredictionInvoker {
  
  // MARK: Model parameters
  let batchSize = 28
  let inputChannels = 3
  
  // image mean and std for floating model, should be consistent with parameters used in model training
  var imageMean: Float = 0
  var imageStd:  Float = 0
  
  private var interpreter: Interpreter
  private let inputHeight: Int
  private let inputWidth: Int
  
  init(interpreter: Interpreter, inputHeight: Int, inputWidth: Int) {
    self.interpreter = interpreter
    self.inputHeight = inputHeight
    self.inputWidth = inputWidth
  }
  
  func invoke(scaledPixelBuffer: CVPixelBuffer) -> Tensor? {
    do {
      let inputTensor = try interpreter.input(at: 0)
      
      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        scaledPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }
      
      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)
      
      // Run inference by invoking the `Interpreter`.
      try interpreter.invoke()
      
      
      return try interpreter.output(at: 0)
      
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }
  }
  
  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The BGRA pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let destinationChannelCount = 3
    let destinationBytesPerRow = destinationChannelCount * width
    
    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      print("Error: out of memory")
      return nil
    }
    
    defer {
      free(destinationData)
    }
    
    var destinationBuffer = vImage_Buffer(data: destinationData,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: destinationBytesPerRow)
    
    if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
      vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
      vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    }
    
    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
    if isModelQuantized {
      return byteData
    }
    
    let floatsByteData = byteData.asFloats()
    calculateMeanAndStdFromFloats(floats: floatsByteData)
    
    // Not quantized, convert to floats
    var floats = [Float]()
    for i in 0..<floatsByteData.count {
      floats.append((floatsByteData[i] - imageMean) / imageStd)
    }
    return Data(copyingBufferOf: floats)
  }
  
  private func calculateMeanAndStdFromFloats(floats: [Float]) {
    let stride = vDSP_Stride(1)
    let n = vDSP_Length(floats.count)
    
    vDSP_normalize(floats, stride, nil, stride, &imageMean, &imageStd, n)
  }
  
}
