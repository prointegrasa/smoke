//
//  FuelPostProcessor.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 07/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import TensorFlowLite

struct FuelPostProcessor: PostProcessor {
  func invoke(modelOut: Tensor) -> Any? {
    guard let outData = [Float](unsafeData: modelOut.data) else {
      return nil
    }
    
    let badFuelConfidence = outData[0]
    let goodFuelConfidence = outData[1]
    
    let isGoodFuel = goodFuelConfidence > badFuelConfidence
    let label = isGoodFuel ? "good fuel" : "bad fuel"
    let confidence = isGoodFuel ? goodFuelConfidence : badFuelConfidence
    
    return "\(label): " + String(format: "%.2f", confidence * 100)
  }
  
  
}
