//
//  BoxTransformationProvider.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 07/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import Foundation
import Matft

protocol BoxTransformationProvider {
  func transform(box: PredictionBoxes) -> PredictionBoxes?
}

struct BoxTransformation: BoxTransformationProvider {
  func transform(box: PredictionBoxes) -> PredictionBoxes? {
    return nil
  }
}

struct HorizontalBoxTransformation: BoxTransformationProvider {
  
  func transform(box: PredictionBoxes) -> PredictionBoxes? {
    if (box.boxes.count == 0) {
      return nil
    }
    
    let transformations = CommonBoxTransformations()
    let transformedBoxes = Matft.hstack(transformations.transformPredictionBoxBoxes(box: box)).reshape([box.boxes.count, 4])
    let imgCenter = MfArray([240, 240])
    
    transformedBoxes[0~<, 0] += 2 * (imgCenter[0] - transformedBoxes[0~<, 0])
    transformedBoxes[0~<, 2] += 2 * (imgCenter[1] - transformedBoxes[0~<, 2])
    
    let boxW = Matft.math.abs(transformedBoxes[0~<, 0] - transformedBoxes[0~<, 2])
    transformedBoxes[0~<, 0] -= boxW
    transformedBoxes[0~<, 2] += boxW
    
    
    let finalBoxes = transformedBoxes.toArray().map { b -> MfArray in
      if let intBox = b as? [Int] {
        return MfArray(intBox.map { v in Double(v) })
      }
      if let floatBox = b as? [Float] {
        return MfArray(floatBox.map { v in Double(v) })
      }
      return MfArray(b as! [Double])
    }
    var prediction = box
    prediction.boxes = CommonBoxTransformations().transformBoxesToCXCYWH(boxes: finalBoxes)
    return prediction
  }
}

struct ScaleBoxTransformation: BoxTransformationProvider {
  private let xScale: Float
  private let yScale: Float
  
  init(xScale: Float, yScale: Float) {
    self.xScale = xScale
    self.yScale = yScale
  }
  
  func transform(box: PredictionBoxes) -> PredictionBoxes? {
    if (box.boxes.count == 0) {
      return nil
    }
    
    let transformations = CommonBoxTransformations()
    var transformedBoxes = Matft.hstack(transformations.transformPredictionBoxBoxes(box: box)).reshape([box.boxes.count, 4])
    
    let scaller = MfArray([xScale, yScale, xScale, yScale])
    transformedBoxes[0~<, ~<4] *= scaller
    let clipBox = [0, 0, 1 + 480, 480]
    
    transformedBoxes = transformations.clipBox(boxes: transformedBoxes, clipBox: clipBox)
    
    let finalBoxes = transformedBoxes.toArray().map{ b in MfArray(b as! [Double]) }
    var prediction = box
    prediction.boxes = CommonBoxTransformations().transformBoxesToCXCYWH(boxes: finalBoxes)
    return prediction
  }
}

struct CommonBoxTransformations {
  func transformPredictionBoxBoxes(box: PredictionBoxes) -> [MfArray] {
    return box.boxes
      .map { b in
        let (cx, cy, w, h) = (b[0], b[1], b[2], b[3])
        return Matft.hstack([
          cx - w / 2,
          cy - h / 2,
          cx + w / 2,
          cy + h / 2
        ])
      }
  }
  
  func transformBoxesToCXCYWH(boxes: [MfArray]) -> [MfArray] {
    return boxes
      .map{b in b.data}
      .map {b in
        if let box = b as? [Double] {
          let (xmin, ymin, xmax, ymax) = (box[0], box[1], box[2], box[3])
          
          let width = xmax - xmin
          let height = ymax - ymin
          
          return MfArray([
            Int(floor(xmin + (width / 2))),
            Int(floor(ymin + (height / 2))),
            Int(floor(width)),
            Int(floor(height))
          ])
        }
        return nil
      }
      .compactMap{ $0 }
  }
  
  func clipBox(boxes: MfArray, clipBox: [Int], alpha: Float = 0.95) -> MfArray {
    let clipBoxArr = MfArray(clipBox)
    
    let xmin = Matft.stats.maximum(boxes[0~<, 0], clipBoxArr[0]).reshape([-1, 1])
    let ymin = Matft.stats.maximum(boxes[0~<, 1], clipBoxArr[1]).reshape([-1, 1])
    let xmax = Matft.stats.minimum(boxes[0~<, 2], clipBoxArr[2]).reshape([-1, 1])
    let ymax = Matft.stats.minimum(boxes[0~<, 3], clipBoxArr[3]).reshape([-1, 1])
    
    let boxArea = bboxArea(boxes: boxes)
    let bbox = Matft.hstack([xmin, ymin, xmax, ymax, boxes[0~<, 4~<]])
    let deltaArea = (boxArea - bboxArea(boxes: bbox)) / boxArea
    let mask = Matft.less(deltaArea, 1 - alpha).astype(.Int)
    if mask.shape[0] == 1 {
      return bbox
    }
    return bbox[mask === 1]
  }
  
  func bboxArea(boxes: MfArray) -> MfArray {
    return (boxes[0~<, 2] - boxes[0~<, 0]) * (boxes[0~<, 3] - boxes[0~<, 1])
  }
}
