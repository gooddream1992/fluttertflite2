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
// Parts of the code are under this licence


import Flutter
import UIKit


public class SwiftFlutterTflitePlugin: NSObject, FlutterPlugin {
  var detector: Yolov4Classifier? = nil
  var registrar: FlutterPluginRegistrar? = nil
  var modelLoaded: Bool = false

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "com.things/flutter_tflite", binaryMessenger: registrar.messenger())
    let instance = SwiftFlutterTflitePlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  init(of registrar: FlutterPluginRegistrar) {
        super.init()
        self.registrar = registrar
  }


  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) 
  {
    switch call.method {
    case "getPlatformVersion":
      result("iOS " + UIDevice.current.systemVersion)
    case "loadModel":
      loadModel(args: call.arguments as? Dictionary<String, Any>, result: result)
    case "detectObjectOnImage":
      detectObjectOnImage(args: call.arguments as? Dictionary<String, Any>, result: result)
    case "detectObjectOnFrame":
      detectObjectOnFrame(args: call.arguments as? Dictionary<String, Any>, result: result)
    case "close":

    default:
      result.notImplemented()
    }   
    
  }

  func loadModel(args: Dictionary<String, Any>, result: @escaping FlutterResult)
  {
    do{
        let modelFilePath = (args!["modelFilePath"] as? String)!
        let labelsFilePath = (args!["labelsFilePath"] as? String)!

        guard let labels = args!["labels"] as? String else { return result("No labels provided") }
        guard let key = registrar?.lookupKey(forAsset: modelPath) else { return result("No model path provided") }
    } 
    catch let error as NSError {
        result.error("", "Failed to load model. " + error.localizedDescription, error)
    }  

  }

  func detectObjectOnImage(args: Dictionary<String, Any>, result: @escaping FlutterResult)
  {
    do{
        guard let imagePath = (args!["imagePath"] as? String) else { return result("No image path provided") }      
    }
    catch let error as NSError {
        result.error("", "Failed to load model. " + error.localizedDescription, error)
    } 
  }

  func detectObjectOnFrame(args: Dictionary<String, Any>, result: @escaping FlutterResult)
  {
    do{
        guard let path = (args!["image"] as? String) else { return result("No image path provided") }      
    }
    catch let error as NSError {
        result.error("", "Failed to load model. " + error.localizedDescription, error)
    } 
  }

  func close(result: @escaping FlutterResult)
  {
      detector.close()
      result("Closed.")
  }
}
