//
//  IndicesOperationsService.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 03/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import Foundation
import Matft

struct IndicesOperations {
    static func getForIndices(arr: MfArray, indices: MfArray) -> MfArray {
        if (indices.shape.count == 0) {
            return MfArray([])
        }
        
        // let wasMatrix = arr.shape.count > 1
        var array: MfArray = arr
        if (arr.shape.count == 1) {
            array = arr.reshape([arr.count, 1])
        }
        array = array[indices]
//        if (wasMatrix) {
//            array = array.reshape([1, array[0].count])
//        } else if (array.shape.count == 1 || array.shape.count == 2 && array[1].count == 1) {
//            array = array.reshape([array[0].count])
//        }
        return array
    }
}
