//
//  SmokeDetectionModels.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 07/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import UIKit

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
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
