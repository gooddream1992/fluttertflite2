package com.things.flutter_tflite

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.renderscript.*
import android.view.Surface
import androidx.annotation.NonNull
import io.flutter.FlutterInjector
import io.flutter.embedding.engine.loader.FlutterLoader
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.serialization.encodeToString
import java.io.IOException
import java.nio.ByteBuffer
import java.util.*
import kotlinx.serialization.json.*


/** FlutterTflitePlugin */
class FlutterTflitePlugin: FlutterPlugin, MethodCallHandler {
  /// The MethodChannel that will the communication between Flutter and native Android
  ///
  /// This local reference serves to register the plugin with the Flutter Engine and unregister it
  /// when the Flutter Engine is detached from the Activity
  private lateinit var channel : MethodChannel
  private lateinit var context: Context
  private lateinit var assetManager: AssetManager
  private lateinit var flutterLoader: FlutterLoader
  private var objectDetector: ObjectDetector? = null
  
  val TF_OD_API_INPUT_SIZE = 416
  private val TF_OD_API_IS_QUANTIZED = false
  
  override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
    context = flutterPluginBinding.applicationContext
    assetManager = flutterPluginBinding.applicationContext.assets
    flutterLoader = FlutterInjector.instance().flutterLoader()
    channel = MethodChannel(flutterPluginBinding.binaryMessenger, "com.things/flutter_tflite")
    channel.setMethodCallHandler(this)
  }


  override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
    channel.setMethodCallHandler(null)
  }


  override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
    when (call.method) {
      "getPlatformVersion" -> result.success("Android ${android.os.Build.VERSION.RELEASE}")
      "loadModel" -> {
        try {
          loadModel(call.arguments as HashMap<String, *>, result)
        } catch (e: Exception) {
          result.error("", "Failed to load model. " + e.message, e)
        }
      }
      "detectObjectOnImage" -> {
        try {
          detectObjectOnImage(call.arguments as HashMap<String, *>, result)
        } catch (e: Exception) {
          result.error("", "Failed to run model on image. " + e.message, e)
        }
      }
      "detectObjectOnFrame" -> {
        try {
          detectObjectOnFrame(call.arguments as HashMap<String, *>, result)
        } catch (e: Exception) {
          result.error("", "Failed to run model on frame. " + e.message, e)
        }
      }
      "close" -> {
        try {
          close(result)
        } catch (e: Exception) {
          result.error("", "Failed to close object detector. " + e.message, e)
        }
      }
      else -> {
        result.notImplemented()
      }
    }
  }


  @kotlin.Throws(IOException::class)
  fun loadModel(args: HashMap<String, *>, result: Result) {
    val modelFilePath = args["modelFilePath"] as String
    val labelsFilePath = args["labelsFilePath"] as String
    val isQuantized = (args["isQuantized"] ?: TF_OD_API_IS_QUANTIZED) as Boolean
    val isTinyYolo = (args["isTinyYolo"] ?: true) as Boolean
    val useGPU = (args["useGPU"] ?: true) as Boolean
    val useNNAPI = (args["useNNAPI"] ?: false) as Boolean
    val isAsset = (args["isAsset"] ?: false) as Boolean

    val actualModelPath = if (isAsset) flutterLoader.getLookupKeyForAsset(modelFilePath) else modelFilePath
    val actualLabelsPath = flutterLoader.getLookupKeyForAsset(labelsFilePath)
    objectDetector = YoloV4.create(assetManager, actualModelPath, actualLabelsPath, isQuantized, isTinyYolo, useGPU, useNNAPI, isAsset)
    result.success("Model $modelFilePath loaded.")
  }


  @kotlin.Throws(IOException::class)
  fun detectObjectOnImage(args: HashMap<String, *>, result: Result) {
    val imagePath = (args["imagePath"] ?:"assets/kite.jpg") as String
    val actualImagePath = flutterLoader.getLookupKeyForAsset(imagePath)

    var sourceBitmap = Utils.getBitmapFromAsset(assetManager, actualImagePath);
    if (sourceBitmap != null) {
      var cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);
      val results = objectDetector?.detectOnImage(cropBitmap)
      result.success(Json.encodeToString(results))
    }
    else
    {
      result.error("ErrorLoadingImage", "Could not load image $imagePath.", "")
    }
  }


  @kotlin.Throws(IOException::class)
  fun detectObjectOnFrame(args: HashMap<String, *>, result: Result) {
    val bytesList: List<ByteArray>  = args["bytesList"] as List<ByteArray>
    //val strides: IntArray = args["strides"] as IntArray
    val width: Int = args["width"] as Int
    val height: Int = args["height"] as Int
    //val compress: Boolean = args["compress"] as Boolean
    val rotation: Int = args["rotation"] as Int

    val bitmap = getBitmapFromFrame(bytesList, width, height, rotation)    
    if (bitmap != null) {
      var cropBitmap = Utils.processBitmap(bitmap, TF_OD_API_INPUT_SIZE);
      val results = objectDetector?.detectOnImage(cropBitmap)
      result.success(Json.encodeToString(results))
    }
    else
    {
      result.error("NullBitmap", "The input frame could not be read.", "")
    }
  }


  fun close(result: Result) {
    objectDetector?.close()
    result.success("Closed.")
  }


  @kotlin.Throws(IOException::class)
  private fun getBitmapFromFrame(bytesList: List<ByteArray>, width: Int, height: Int, rotation: Int): Bitmap? {
    val Y = ByteBuffer.wrap(bytesList[0])
    val U = ByteBuffer.wrap(bytesList[1])
    val V = ByteBuffer.wrap(bytesList[2])

    val Yb = Y.remaining()
    val Ub = U.remaining()
    val Vb = V.remaining()

    val data = ByteArray(Yb + Ub + Vb)

    Y[data, 0, Yb]
    V[data, Yb, Vb]
    U[data, Yb + Vb, Ub]

    var bitmapRaw = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val bmData = renderScriptNV21ToRGBA888(data, width, height)
    bmData!!.copyTo(bitmapRaw)

    val matrix = Matrix()
    matrix.postRotate(rotation.toFloat())
    bitmapRaw = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.width, bitmapRaw.height, matrix, true)

    return bitmapRaw
  }


  private fun renderScriptNV21ToRGBA888(nv21: ByteArray, width: Int, height: Int): Allocation? {
    // https://stackoverflow.com/a/36409748
    val rs = RenderScript.create(context)
    val yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
    val yuvType: Type.Builder = Type.Builder(rs, Element.U8(rs)).setX(nv21.size)
    val input = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT)
    val rgbaType: Type.Builder = Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height)
    val out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT)
    input.copyFrom(nv21)
    yuvToRgbIntrinsic.setInput(input)
    yuvToRgbIntrinsic.forEach(out)
    return out
  }  

}
