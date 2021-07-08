//
//  Prediction.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 04/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import Foundation
import Matft

struct PredictionBoxes {
  public var boxes: [MfArray]
  public var probabilities: [Float]
  public var classIndexes: [Int]
  public var isAugmented: Bool
  
  init(boxes: [MfArray], probabilities: [Float], classIndexes: [Int], isAugmented: Bool = false) {
    self.boxes = boxes
    self.probabilities = probabilities
    self.classIndexes = classIndexes
    self.isAugmented = isAugmented
  }
}
