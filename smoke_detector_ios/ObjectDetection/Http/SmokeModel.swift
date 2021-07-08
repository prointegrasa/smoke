//
//  SmokeModel.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 07/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import Foundation

struct SmokePrediction: Encodable {
  let created_date: String
  let user_id: String
  let frame: String? // frameSize
  let crop: String? // cropSize
  let inference_time: String? // ms
  let final_threshold: String
  let gpu: String
  let tta: String
  let jpg: String? // base64
  let danger_degree: String?
  let location: String
}
