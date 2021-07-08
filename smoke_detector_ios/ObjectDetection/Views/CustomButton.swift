//
//  CustomButton.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 08/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import UIKit

class CustomButton: UIButton {
    enum ButtonState {
        case normal
        case disabled
    }

    private var disabledBackgroundColor: UIColor?
    private var defaultBackgroundColor: UIColor? {
        didSet {
            backgroundColor = defaultBackgroundColor
        }
    }
    
    override var isEnabled: Bool {
        didSet {
            if isEnabled {
                if let color = defaultBackgroundColor {
                    self.backgroundColor = color
                }
            }
            else {
                if let color = disabledBackgroundColor {
                    self.backgroundColor = color
                }
            }
        }
    }
    
    func setBackgroundColor(_ color: UIColor?, for state: ButtonState) {
        switch state {
        case .disabled:
            disabledBackgroundColor = color
        case .normal:
            defaultBackgroundColor = color
        }
    }
}
