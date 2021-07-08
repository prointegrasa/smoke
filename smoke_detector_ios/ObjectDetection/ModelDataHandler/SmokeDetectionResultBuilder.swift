//
//  SmokeDetectionResultBuilder.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 07/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import TensorFlowLite

struct SmokeDetectionResultBuilder: ResultBuilder {
  
  private let postProcess: ModelPostProcessor
  
  init(_ postProcess: ModelPostProcessor) {
    self.postProcess = postProcess
  }
  
  func build(modelOut: Tensor) -> AnyObject? {
    var inferences: [Inference] = []
    guard let prediction = postProcess.invoke(modelOut: modelOut) else {
      return nil
    }
    var bestPrediction = prediction
    if ttaEnabled && !continuesDetection {
      let ttaService = TTAService(predictionInvoker: predictionInvoker, postProcessor: postProcess)
      if let foundPrediction = ttaService.findBetterPrediction(forImage: scaledPixelBuffer, withPrediction: prediction) {
        bestPrediction = foundPrediction
      }
    }
    
    bestPrediction = rearrangeBoxes(onPrediction: bestPrediction)
    inferences = piResultToApplication(pred: bestPrediction, imageWidth: imageWidth, imageHeight: imageHeight)
    
    // Returns the inference time and inferences
    let result = Result(inferenceTime: interval, inferences: inferences, scaledBuffer: scaledPixelBuffer)
    return result
  }
  
}
