// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit
import CoreLocation

public class DetectionRunning {

    private let lock = DispatchSemaphore(value: 1)
    private var value = false

    // You need to lock on the value when reading it too since
    // there are no volatile variables in Swift as of today.
    public func get() -> Bool {

        lock.wait()
        defer { lock.signal() }
        return value
    }

    public func set(_ newValue: Bool) {

        lock.wait()
        defer { lock.signal() }
        value = newValue
    }
}

class ViewController: UIViewController, CLLocationManagerDelegate {
  
  // MARK: Storyboards Connections
  @IBOutlet weak var previewView: PreviewView!
  @IBOutlet weak var overlayView: OverlayView!
  @IBOutlet weak var resumeButton: UIButton!
  @IBOutlet weak var continuesDetectionButton: UIButton!
  @IBOutlet weak var shooterButton: UIButton!
  @IBOutlet weak var cameraUnavailableLabel: UILabel!
  @IBOutlet weak var generatingResultsLabel: UILabel!
  @IBOutlet weak var fuelClassificationLabel: UILabel!
  
  @IBOutlet weak var bottomSheetStateImageView: UIImageView!
  @IBOutlet weak var bottomSheetView: UIView!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var bottomSheetViewBottomSpace: NSLayoutConstraint!
  
  // MARK: Constants
  private let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)
  private let edgeOffset: CGFloat = 2.0
  private let labelOffset: CGFloat = 10.0
  private let animationDuration = 0.5
  private let collapseTransitionThreshold: CGFloat = -30.0
  private let expandTransitionThreshold: CGFloat = 30.0
  private let delayBetweenInferencesMs: Double = 200
  private var continuesDetection: Bool = false
  private var wasContinuesSelected: Bool = false
  private var shooterClicked: Bool = false
  private var resultImage: UIImage? = nil
  private var frameImage: UIImage? = nil
  private var isTTAEnabled: Bool = true
  private var threshold: Double = 0.6
  private var restoreContinuesDetection: Bool = false
  private var isDetectionRunning: DetectionRunning = DetectionRunning()
  // private var continuesDetectionDisabled: DetectionRunning = DetectionRunning()
  
  // MARK: Instance Variables
  private var initialBottomSpace: CGFloat = 0.0
  private let locationManager: CLLocationManager = CLLocationManager()
  
  private var location: CLLocation? = nil
  
  // Holds the results at any time
  private var result: Result?
  private var previousInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 1000
  
  // MARK: Controllers that manage functionality
  private lazy var cameraFeedManager = CameraFeedManager(previewView: previewView)
  
  private var selectedDataHandler: ModelDataHandler?
  public var modelDataHandler: ModelDataHandler? {
    get {
      return selectedDataHandler
    }
    set {
      selectedDataHandler = newValue
      isSmokeDetection = selectedDataHandler?.isSmokeDetectionModel
    }
  }
  
  private var inferenceViewController: InferenceViewController?
  private var isSmokeDetection: Bool!
  
  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()
    
    guard modelDataHandler != nil else {
      fatalError("Failed to load model")
    }
    cameraFeedManager.delegate = self
    overlayView.clearsContextBeforeDrawing = true
    
    addPanGesture()
    
    shooterButton.isHidden = !isSmokeDetection
    fuelClassificationLabel.isHidden = isSmokeDetection
    continuesDetection = !isSmokeDetection
    continuesDetectionButton.isHidden = !isSmokeDetection
  }
  
  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // Dispose of any resources that can be recreated.
  }
  
  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    locationManager.delegate = self
    
    locationManager.requestWhenInUseAuthorization()
    locationManager.startUpdatingLocation()
    
    if resultImage == nil {
      changeBottomViewState()
    } else {
      restartUI()
    }
    cameraFeedManager.checkCameraConfigurationAndStartSession()
  }
  
  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)
    
    cameraFeedManager.stopSession()
  }
  
  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }
  
  // MARK: Button Actions
  @IBAction func onClickResumeButton(_ sender: Any) {
    
    cameraFeedManager.resumeInterruptedSession { (complete) in
      
      if complete {
        self.resumeButton.isHidden = true
        self.cameraUnavailableLabel.isHidden = true
      }
      else {
        self.presentUnableToResumeSessionAlert()
      }
    }
  }
  
  @IBAction func finishVIew(sender: AnyObject) {
    self.dismiss(animated: true, completion: nil)
  }
  
  @IBAction func continuesDetectionButton(sender: AnyObject) {
    self.continuesDetection = !self.continuesDetection
    self.continuesDetectionButton.isSelected = self.continuesDetection
    self.restoreContinuesDetection = continuesDetection
    self.wasContinuesSelected = continuesDetectionButton.isSelected
    
//    if !continuesDetection {
//      continuesDetectionDisabled.set(true)
//    } else if continuesDetectionDisabled.get() {
//      continuesDetectionDisabled.set(false)
//    }
    
    self.inferenceViewController?.changeTTAButtonStateDueToContinuesDetection()
    
    self.draw(objectOverlays: [])
  }
  
  @IBAction func shooterButtonClicked(sender: AnyObject) {
    self.shooterClicked = true
//    self.continuesDetectionDisabled.set(false)
    self.continuesDetection = true
    self.generatingResultsLabel.isHidden = false
    self.activityIndicator.isHidden = false
    self.shooterButton.isHidden = true
    self.continuesDetectionButton.isHidden = true
  }
  
  private func restartUI() {
    self.shooterClicked = false
    self.continuesDetection = self.restoreContinuesDetection
    self.generatingResultsLabel.isHidden = true
    self.activityIndicator.isHidden = true
    self.shooterButton.isHidden = false
    self.continuesDetectionButton.isHidden = false
    self.continuesDetectionButton.isSelected = self.continuesDetection
    
    if continuesDetection {
      DispatchQueue.main.asyncAfter(deadline: .now() + (isTTAEnabled ? 1.0 : 2.0)) {
        self.isDetectionRunning.set(false)
      }
    } else {
      isDetectionRunning.set(false)
    }
    
    self.draw(objectOverlays: [])
    cameraFeedManager.checkCameraConfigurationAndStartSession()
  }
  
  func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
    locationManager.stopUpdatingLocation()
    if let location = locations.first {
      self.location = location
    }
  }
  
  func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
      // Handle failure to get a user’s location
    print("Location error \(error.localizedDescription)")
  }
  
  func presentUnableToResumeSessionAlert() {
    let alert = UIAlertController(
      title: "Unable to Resume Session",
      message: "There was an error while attempting to resume session.",
      preferredStyle: .alert
    )
    alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
    
    self.present(alert, animated: true)
  }
  
  // MARK: Storyboard Segue Handlers
  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    super.prepare(for: segue, sender: sender)
    
    if segue.identifier == "EMBED" {
      
      guard let tempModelDataHandler = modelDataHandler else {
        return
      }
      inferenceViewController = segue.destination as? InferenceViewController
      inferenceViewController?.wantedInputHeight = tempModelDataHandler.inputHeight
      inferenceViewController?.wantedInputWidth = tempModelDataHandler.inputWidth
      inferenceViewController?.threadCountLimit = tempModelDataHandler.threadCountLimit
      inferenceViewController?.currentThreadCount = tempModelDataHandler.threadCount
      inferenceViewController?.delegate = self
      
      guard let tempResult = result else {
        return
      }
      inferenceViewController?.inferenceTime = tempResult.inferenceTime
      
    }
    if segue.identifier == "ResultsShowSegue" {
      let resultsViewController = segue.destination as? ResultsViewController
      _ = resultsViewController?.view
      guard let tempImage = resultImage else {
        return
      }
      guard let loc = location else {
        return
      }
      
      resultsViewController?.resultImage = tempImage
      if let detectionCount = result?.inferences.count {
        if detectionCount > 0 {
          resultsViewController?.foundResults = true
          resultsViewController?.frameImage = frameImage
          resultsViewController?.frameImage = frameImage
          resultsViewController?.latitude = loc.coordinate.latitude
          resultsViewController?.longitude = loc.coordinate.longitude
          resultsViewController?.inferenceTime = result?.inferenceTime
          resultsViewController?.threshold = threshold
          resultsViewController?.ttaEnabled = isTTAEnabled ? "1" : "0"
          resultsViewController?.dangerDegree = result?.inferences[0].className.replacingOccurrences(of: "Danger: ", with: "")
        }
      }
    }
  }
}

// MARK: InferenceViewControllerDelegate Methods
extension ViewController: InferenceViewControllerDelegate {
  
  func didChangeThreadCount(to count: Int) {
    if modelDataHandler?.threadCount == count { return }
    modelDataHandler = ModelDataHandler(
      modelFileInfo: modelDataHandler!.modelFile,
      postProcessor: modelDataHandler!.postProcess,
      threadCount: count,
      inputWidth: modelDataHandler!.inputWidth,
      inputHeight: modelDataHandler!.inputHeight
    )
  }
  
  func didChangeTTAValue(to isEnabled: Bool) {
    isTTAEnabled = isEnabled
  }
  
  func didChangeThresholdValue(to value: Double) {
    threshold = value
  }
  
}

// MARK: CameraFeedManagerDelegate Methods
extension ViewController: CameraFeedManagerDelegate {
  
  func didOutput(pixelBuffer: CVPixelBuffer) {
    if (self.continuesDetection) {
      runModel(onPixelBuffer: pixelBuffer)
    }
  }
  
  // MARK: Session Handling Alerts
  func sessionRunTimeErrorOccurred() {
    
    // Handles session run time error by updating the UI and providing a button if session can be manually resumed.
    self.resumeButton.isHidden = false
  }
  
  func sessionWasInterrupted(canResumeManually resumeManually: Bool) {
    
    // Updates the UI when session is interrupted.
    if resumeManually {
      self.resumeButton.isHidden = false
    }
    else {
      self.cameraUnavailableLabel.isHidden = false
    }
  }
  
  func sessionInterruptionEnded() {
    
    // Updates UI once session interruption has ended.
    if !self.cameraUnavailableLabel.isHidden {
      self.cameraUnavailableLabel.isHidden = true
    }
    
    if !self.resumeButton.isHidden {
      self.resumeButton.isHidden = true
    }
  }
  
  func presentVideoConfigurationErrorAlert() {
    
    let alertController = UIAlertController(title: "Configuration Failed", message: "Configuration of camera has failed.", preferredStyle: .alert)
    let okAction = UIAlertAction(title: "OK", style: .cancel, handler: nil)
    alertController.addAction(okAction)
    
    present(alertController, animated: true, completion: nil)
  }
  
  func presentCameraPermissionsDeniedAlert() {
    
    let alertController = UIAlertController(title: "Camera Permissions Denied", message: "Camera permissions have been denied for this app. You can change this by going to Settings", preferredStyle: .alert)
    
    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
      
      UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
    }
    
    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)
    
    present(alertController, animated: true, completion: nil)
    
  }
  
  /** This method runs the live camera pixelBuffer through tensorFlow to get the result.
   */
  @objc  func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer) {
    if isDetectionRunning.get() {
      return
    }
    isDetectionRunning.set(true)
    // Run the live camera pixelBuffer through tensorFlow to get the result
    
    let currentTimeMs = Date().timeIntervalSince1970 * 1000
    
    guard  (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs || (shooterClicked && !continuesDetection) else {
      return
    }
    
    if (shooterClicked) { // we allow to click shooter only once
      continuesDetection = false
      cameraFeedManager.stopSession()
      wasContinuesSelected = continuesDetectionButton.isSelected
    }
    
//    guard let url = Bundle.main.url(
//      forResource: "image2",
//      withExtension: "jpg"
//    ) else {
//      print("Failed to load the image file with name: sample.")
//      return
//    }
//
//    do {
//      let imageData = try Data(contentsOf: url)
//      if let img = UIImage(data: imageData) {
//        if let ciImage = CIImage(image: img) {
//          let context = CIContext()
//          context.render(ciImage, to: pixelBuffer)
//        }
//      }
//    } catch {
//      print("Error loading image : \(error)")
//    }
    
    previousInferenceTimeMs = currentTimeMs
    self.modelDataHandler?.threshold = Float(threshold)
    self.modelDataHandler?.ttaEnabled = isTTAEnabled
    
    result = self.modelDataHandler?.runModel(onFrame: pixelBuffer)
    
    guard let displayResult = result else {
      return
    }
    
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    
    DispatchQueue.main.async {
      
      // Display results by handing off to the InferenceViewController
      self.inferenceViewController?.resolution = CGSize(width: width, height: height)
      
      if (self.shooterClicked || (self.continuesDetection && displayResult.inferences.count > 0)) {
        if self.isSmokeDetection {
          self.drawAfterPerformingCalculations(onInferences: displayResult.inferences, withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)))
          self.generateImageToSendAndShowItToUser(pixelBuffer: pixelBuffer, scaledPixelBuffer: displayResult.scaledBuffer)
        } else {
          self.setFuelClassificationlabel(label: displayResult.label)
        }
      } else {
        self.isDetectionRunning.set(false)
        var inferenceTime: Double = 0
        if let resultInferenceTime = self.result?.inferenceTime {
          inferenceTime = resultInferenceTime
        }
        self.inferenceViewController?.inferenceTime = inferenceTime
        self.inferenceViewController?.tableView.reloadData()
        
        if self.isSmokeDetection {
          // Draws the bounding boxes and displays class names and confidence scores.
          self.drawAfterPerformingCalculations(onInferences: displayResult.inferences, withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)))
        } else {
          self.setFuelClassificationlabel(label: displayResult.label)
        }
      }
    }
  }
  
  private func setFuelClassificationlabel(label: String?) {
    if let text = label {
      self.fuelClassificationLabel.text = text
    }
  }
  
  /**
   This method takes the results, translates the bounding box rects to the current view, draws the bounding boxes, classNames and confidence scores of inferences.
   */
  func drawAfterPerformingCalculations(onInferences inferences: [Inference], withImageSize imageSize:CGSize) {
    
    self.overlayView.objectOverlays = []
    self.overlayView.setNeedsDisplay()
    
    guard !inferences.isEmpty else {
      return
    }
    
    var objectOverlays: [ObjectOverlay] = []
    
    for inference in inferences {
      
      // Translates bounding box rect to current view.
      var convertedRect = inference.rect.applying(CGAffineTransform(scaleX: self.overlayView.bounds.size.width / imageSize.width, y: self.overlayView.bounds.size.height / imageSize.height))
      
      if convertedRect.origin.x < 0 {
        convertedRect.origin.x = self.edgeOffset
      }
      
      if convertedRect.origin.y < 0 {
        convertedRect.origin.y = self.edgeOffset
      }
      
      if convertedRect.maxY > self.overlayView.bounds.maxY {
        convertedRect.size.height = self.overlayView.bounds.maxY - convertedRect.origin.y - self.edgeOffset
      }
      
      if convertedRect.maxX > self.overlayView.bounds.maxX {
        convertedRect.size.width = self.overlayView.bounds.maxX - convertedRect.origin.x - self.edgeOffset
      }
      
      let confidenceValue = Int(inference.confidence * 100.0)
      let string = "\(inference.className)  (\(confidenceValue)%)"
      
      let size = string.size(usingFont: self.displayFont)
      
      let objectOverlay = ObjectOverlay(name: string, borderRect: convertedRect, nameStringSize: size, color: inference.displayColor, font: self.displayFont)
      
      objectOverlays.append(objectOverlay)
    }
    
    if (self.continuesDetection || self.shooterClicked) {
      // Hands off drawing to the OverlayView
      self.draw(objectOverlays: objectOverlays)
    }
    
  }
  
  /** Calls methods to update overlay view with detected bounding boxes and class names.
   */
  func draw(objectOverlays: [ObjectOverlay]) {
    
    self.overlayView.objectOverlays = objectOverlays
    self.overlayView.setNeedsDisplay()
  }
  
  func generateImageToSendAndShowItToUser(pixelBuffer: CVPixelBuffer, scaledPixelBuffer: CVPixelBuffer) {
    let previewImage = scaledPixelBuffer.asImage()
    var overlayImage = self.overlayView.asImage()
    // match size of camera preview for which overlay is generated
    let resizedOverlay = overlayImage.resizeImage(targetSize: CGSize(width: 1080, height: 1920))
    guard let cropped = resizedOverlay.cgImage?.cropping(to: CGRect(x: 0, y: 420, width: 1080, height: 1080)) else {
      return
    }
    overlayImage = UIImage(cgImage: cropped).resizeImage(targetSize: previewImage.size)
    
    self.frameImage = pixelBuffer.asImage()
    self.resultImage = previewImage.overlayWith(image: overlayImage, posX: 0, posY: 0).resizeImage(targetSize: CGSize(width: 480, height: 480))
    guard self.continuesDetectionButton.isSelected == self.wasContinuesSelected else {
      print("state differs")
      draw(objectOverlays: [])
      return
    }
    
    if let _ = location {
      performSegue(withIdentifier: "ResultsShowSegue", sender: nil)
    }
  }
  
}

// MARK: Bottom Sheet Interaction Methods
extension ViewController {
  
  // MARK: Bottom Sheet Interaction Methods
  /**
   This method adds a pan gesture to make the bottom sheet interactive.
   */
  private func addPanGesture() {
    let panGesture = UIPanGestureRecognizer(target: self, action: #selector(ViewController.didPan(panGesture:)))
    bottomSheetView.addGestureRecognizer(panGesture)
  }
  
  
  /** Change whether bottom sheet should be in expanded or collapsed state.
   */
  private func changeBottomViewState() {
    
    guard let inferenceVC = inferenceViewController else {
      return
    }
    
    if bottomSheetViewBottomSpace.constant == inferenceVC.collapsedHeight - bottomSheetView.bounds.size.height {
      
      bottomSheetViewBottomSpace.constant = 0.0
    }
    else {
      bottomSheetViewBottomSpace.constant = inferenceVC.collapsedHeight - bottomSheetView.bounds.size.height
    }
    setImageBasedOnBottomViewState()
  }
  
  /**
   Set image of the bottom sheet icon based on whether it is expanded or collapsed
   */
  private func setImageBasedOnBottomViewState() {
    
    if bottomSheetViewBottomSpace.constant == 0.0 {
      bottomSheetStateImageView.image = UIImage(named: "down_icon")
    }
    else {
      bottomSheetStateImageView.image = UIImage(named: "up_icon")
    }
  }
  
  /**
   This method responds to the user panning on the bottom sheet.
   */
  @objc func didPan(panGesture: UIPanGestureRecognizer) {
    
    // Opens or closes the bottom sheet based on the user's interaction with the bottom sheet.
    let translation = panGesture.translation(in: view)
    
    switch panGesture.state {
    case .began:
      initialBottomSpace = bottomSheetViewBottomSpace.constant
      translateBottomSheet(withVerticalTranslation: translation.y)
    case .changed:
      translateBottomSheet(withVerticalTranslation: translation.y)
    case .cancelled:
      setBottomSheetLayout(withBottomSpace: initialBottomSpace)
    case .ended:
      translateBottomSheetAtEndOfPan(withVerticalTranslation: translation.y)
      setImageBasedOnBottomViewState()
      initialBottomSpace = 0.0
    default:
      break
    }
  }
  
  /**
   This method sets bottom sheet translation while pan gesture state is continuously changing.
   */
  private func translateBottomSheet(withVerticalTranslation verticalTranslation: CGFloat) {
    
    let bottomSpace = initialBottomSpace - verticalTranslation
    guard bottomSpace <= 0.0 && bottomSpace >= inferenceViewController!.collapsedHeight - bottomSheetView.bounds.size.height else {
      return
    }
    setBottomSheetLayout(withBottomSpace: bottomSpace)
  }
  
  /**
   This method changes bottom sheet state to either fully expanded or closed at the end of pan.
   */
  private func translateBottomSheetAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat) {
    
    // Changes bottom sheet state to either fully open or closed at the end of pan.
    let bottomSpace = bottomSpaceAtEndOfPan(withVerticalTranslation: verticalTranslation)
    setBottomSheetLayout(withBottomSpace: bottomSpace)
  }
  
  /**
   Return the final state of the bottom sheet view (whether fully collapsed or expanded) that is to be retained.
   */
  private func bottomSpaceAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat) -> CGFloat {
    
    // Calculates whether to fully expand or collapse bottom sheet when pan gesture ends.
    var bottomSpace = initialBottomSpace - verticalTranslation
    
    var height: CGFloat = 0.0
    if initialBottomSpace == 0.0 {
      height = bottomSheetView.bounds.size.height
    }
    else {
      height = inferenceViewController!.collapsedHeight
    }
    
    let currentHeight = bottomSheetView.bounds.size.height + bottomSpace
    
    if currentHeight - height <= collapseTransitionThreshold {
      bottomSpace = inferenceViewController!.collapsedHeight - bottomSheetView.bounds.size.height
    }
    else if currentHeight - height >= expandTransitionThreshold {
      bottomSpace = 0.0
    }
    else {
      bottomSpace = initialBottomSpace
    }
    
    return bottomSpace
  }
  
  /**
   This method layouts the change of the bottom space of bottom sheet with respect to the view managed by this controller.
   */
  func setBottomSheetLayout(withBottomSpace bottomSpace: CGFloat) {
    
    view.setNeedsLayout()
    bottomSheetViewBottomSpace.constant = bottomSpace
    view.setNeedsLayout()
  }
  
}

extension CVPixelBuffer {
  func asImage() -> UIImage {
    return UIImage(ciImage: CIImage(cvPixelBuffer: self))
  }
}

extension UIImage {
  
  func overlayWith(image: UIImage, posX: CGFloat, posY: CGFloat) -> UIImage {
      let newWidth = size.width < posX + image.size.width ? posX + image.size.width : size.width
      let newHeight = size.height < posY + image.size.height ? posY + image.size.height : size.height
      let newSize = CGSize(width: newWidth, height: newHeight)

      UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0)
      draw(in: CGRect(origin: CGPoint.zero, size: size))
      image.draw(in: CGRect(origin: CGPoint(x: posX, y: posY), size: newSize))
      let newImage = UIGraphicsGetImageFromCurrentImageContext()!
      UIGraphicsEndImageContext()

      return newImage
    }
  
  func resizeImage(targetSize: CGSize) -> UIImage {
      let size = self.size

      let widthRatio  = targetSize.width  / size.width
      let heightRatio = targetSize.height / size.height

      var newSize: CGSize
      if(widthRatio > heightRatio) {
          newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
      } else {
          newSize = CGSize(width: size.width * widthRatio, height: size.height *      widthRatio)
      }

      let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)

      UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
      self.draw(in: rect)
      let newImage = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext()

      return newImage!
  }
}
