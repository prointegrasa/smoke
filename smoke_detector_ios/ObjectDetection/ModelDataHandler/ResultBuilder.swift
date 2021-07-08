//
//  ResultBuilder.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 07/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

protocol ResultBuilder {
  func build(modelOut: Tensor) -> AnyObject
}
