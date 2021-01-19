import 'dart:async';
import 'dart:typed_data';
import 'package:meta/meta.dart';
import 'package:flutter/services.dart';

class FlutterTflite {
  static const MethodChannel _channel =
      const MethodChannel('com.things/flutter_tflite');

  static Future<String> get platformVersion async {
    final String version = await _channel.invokeMethod('getPlatformVersion');
    return version;
  }

  static Future<String> loadModel(
      {@required String modelFilePath,
      @required String labelsFilePath,
      bool isTinyYolo = true,
      bool useGPU = true,
      bool useNNAPI = false,
      bool isQuantized = false,
      bool isAsset = false}) async {
    return await _channel.invokeMethod(
      'loadModel',
      {
        "modelFilePath": modelFilePath,
        "labelsFilePath": labelsFilePath,
        "isTinyYolo": isTinyYolo,
        'useGPU': useGPU,
        'useNNAPI': useNNAPI,
        "isQuantized": isQuantized,
        "isAsset": isAsset
      },
    );
  }

  static Future<String> detectObjectOnImage(
      {@required String imagePath}) async {
    return await _channel.invokeMethod(
      'detectObjectOnImage',
      {"imagePath": imagePath},
    );
  }

  static Future<String> detectObjectOnFrame({
    @required List<Uint8List> bytesList,
    int imageWidth = 720,
    int imageHeight = 1280,
    int rotation = 90, // Android only
  }) async {
    return await _channel.invokeMethod(
      'detectObjectOnFrame',
      {
        "bytesList": bytesList,
        "width": imageWidth,
        "height": imageHeight,
        "rotation": rotation
      },
    );
  }

  static Future close() async {
    return await _channel.invokeMethod('close');
  }
}
