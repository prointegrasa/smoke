//
//  MainViewController.swift
//  ObjectDetection
//
//  Created by Tomasz Domaracki on 06/06/2021.
//  Copyright © 2021 Y Media Labs. All rights reserved.
//

import UIKit


class MainViewController: UIViewController {
  
  private let smokeDetectionHandler: ModelDataHandler? = ModelDataHandler(modelFileInfo: SmokeDetection.modelInfo, postProcessor: ModelPostProcessor(), inputWidth: 480, inputHeight: 480)
  private let fuelDetectionHandler: ModelDataHandler? = ModelDataHandler(modelFileInfo: FuelDetection.modelInfo, postProcessor: FuelPostProcessor(), inputWidth: 28, inputHeight: 28)
  
  @IBAction func unwindToMainView(segue: UIStoryboardSegue) {

  }
  
  // MARK: Storyboard Segue Handlers
  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    super.prepare(for: segue, sender: sender)
    
    if segue.identifier == "VerifySmokeSegue" {
      let nController = segue.destination as! UINavigationController
      let vController = nController.topViewController as! ViewController
      vController.modelDataHandler = smokeDetectionHandler
    }
    if segue.identifier == "VerifyFuelSegue" {
      let nController = segue.destination as! UINavigationController
      let vController = nController.topViewController as! ViewController
      vController.modelDataHandler = fuelDetectionHandler
    }
  }
}

