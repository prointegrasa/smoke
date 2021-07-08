//
//  SqueezeConfig.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 31/05/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import Foundation
import UIKit
import Matft

struct SqueezeConfig : Decodable {
    let ANCHORS_HEIGHT: Int
    let ANCHORS_WIDTH: Int
    let ANCHOR_PER_GRID: Int
    let ANCHOR_SEED: [[Float]]
    let BATCH_SIZE: Int
    let BATCH_CACHE_SIZE_LIMIT: Int
    let VISUALIZATION_BATCH_SIZE: Int
    let EPOCHS: Int
    let CLASSES: Int
    let CLASS_NAMES: [String]
    let CLASS_TO_IDX: [String: Int]
    let EPSILON: Float
    let EXP_THRESH: Float
    let FINAL_THRESHOLD: Float
    let IMAGE_HEIGHT_PROCESSING: Int
    let IMAGE_WIDTH_PROCESSING: Int
    let IMAGE_HEIGHT_STORAGE: Int
    let IMAGE_WIDTH_STORAGE: Int
    let IOU_THRESHOLD: Float
    let KEEP_PROB: Float
    let LEARNING_RATE: Float
    let LOSS_COEF_BBOX: Float
    let LOSS_COEF_CLASS: Float
    let LOSS_COEF_CONF_NEG: Float
    let LOSS_COEF_CONF_POS: Float
    let MAX_GRAD_NORM: Float
    let MOMENTUM: Float
    let NMS_THRESH: Float
    let N_CHANNELS: Int
    let PROB_THRESH: Float
    let TOP_N_DETECTION: Int
    let WEIGHT_DECAY: Float
    let USE_TTA_ON_PREDICT: Int
    let USE_TTA_ON_EVAL: Int
    let USE_PRED_FINAL_PRESENTATION: Int
    let USE_TTA_MAXIMIZE_PRECISION: Int
    let IMAGE_SERIE: Int
    
    static func fromFile(fileName: String, fileExtension: String) -> SqueezeConfig? {
        guard let configPath = Bundle.main.path(
          forResource: fileName,
          ofType: fileExtension
        ) else {
          print("Failed to load the config file with name: \(fileName).")
          return nil
        }
        
        do {
            let fileContent = try String(contentsOfFile: configPath)
            return fromUtf8String(json: fileContent)
        } catch {
            print("Failed to load the config file with name: \(error.localizedDescription).")
            return nil
        }
    }
    
    static func fromUtf8String(json: String) -> SqueezeConfig? {
        return fromJson(data: Data(json.utf8))
    }
    
    static func fromJson(data: Data) -> SqueezeConfig? {
        var config: SqueezeConfig?
        do {
            config = try JSONDecoder().decode(SqueezeConfig.self, from: data)
        } catch {
            print("Error took place: \(error.localizedDescription).")
        }
        
        return config
    }
    
}

struct SqueezeConfigWithAnchors {
    var anchorBox: MfArray! = nil
    var anchorsCount: Int = 0
    
    let c: SqueezeConfig
    
    init(config: SqueezeConfig) {
        self.c = config
        self.anchorBox = self.generateAnchorBoxFromSeed()
        self.anchorsCount = self.anchorBox.count
    }
    
    private func generateAnchorBoxFromSeed() -> MfArray {
        let H = self.c.ANCHORS_HEIGHT
        let W = self.c.ANCHORS_WIDTH
        let B = self.c.ANCHOR_PER_GRID
        
        let repeatSeedCount = H * W
        let anchorShapes = MfArray(Array(repeating: self.c.ANCHOR_SEED, count: repeatSeedCount))
            .reshape([H, W, B, 2])
        
        let centerXArranged = Matft
            .arange(start: 1, to: W + 1, by: 1)
        let multipliedCenterXArranged = Matft
            .mul(
                centerXArranged,
                Float(self.c.IMAGE_WIDTH_PROCESSING) / Float(W + 1)
            )
            .toArray()
        let centerX = MfArray(Array(repeating: multipliedCenterXArranged, count: H * B))
            .reshape([B, H, W])
            .transpose(axes: [1, 2, 0])
            .reshape([H, W, B, 1])
        
        let centerYArranged = Matft
            .arange(start: 1, to: H + 1, by: 1)
        let multipliedCenterYArranged = Matft
            .mul(
                centerYArranged,
                Float(self.c.IMAGE_HEIGHT_PROCESSING) / Float(H + 1)
            )
            .toArray()
        let centerY = MfArray(Array(repeating: multipliedCenterYArranged, count: W * B))
            .reshape([B, W, H])
            .transpose(axes: [2, 1, 0])
            .reshape([H, W, B, 1])
        
        return Matft
            .concatenate([centerX, centerY, anchorShapes], axis: 3)
            .reshape([-1, 4])
    }
}
