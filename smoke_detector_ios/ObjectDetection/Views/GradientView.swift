//
//  GradientView.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 05/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import Foundation
import UIKit

@IBDesignable
class GradientView: UIView {
  
  override init(frame: CGRect) {
      super.init(frame: frame)
  }
  
  required init?(coder: NSCoder) {
    super.init(coder: coder)
  }
  
  @IBInspectable var firstColor: UIColor = UIColor.clear {
    didSet {
      updateView()
    }
  }
  @IBInspectable var secondColor: UIColor = UIColor.clear {
    didSet {
      updateView()
    }
  }
  
  override class var layerClass: AnyClass {
    get {
      return CAGradientLayer.self
    }
  }
  
  @IBInspectable var isHorizontal: Bool = true {
    didSet {
      updateView()
    }
  }
  
  func updateView() {
    let layer = self.layer as! CAGradientLayer
    layer.colors = [firstColor, secondColor].map{$0.cgColor}
    if (self.isHorizontal) {
      layer.startPoint = CGPoint(x: 0, y: 0.5)
      layer.endPoint = CGPoint (x: 1, y: 0.5)
    } else {
      layer.startPoint = CGPoint(x: 0.5, y: 0)
      layer.endPoint = CGPoint (x: 0.5, y: 1)
    }
  }
}
