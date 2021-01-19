package com.things.flutter_tflite

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.RectF
import android.os.Build
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import kotlin.Comparator
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.math.max
import kotlin.math.min


/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

open class YoloV4 private constructor() : ObjectDetector {
    // tiny or not
    private var isTiny = false
    private var isModelQuantized = false

    override fun enableStatLogging(logStats: Boolean) {}

    override val statString: String
        get() = ""

    override fun close() { tfLite?.close() }
    override fun setNumThreads(num_threads: Int) { tfLite?.setNumThreads(num_threads) }
    override fun setUseNNAPI(isChecked: Boolean) { tfLite?.setUseNNAPI(isChecked) }

    override val objThresh: Float
        get() = 0.5f;


    // Config values.
    // Pre-allocated buffers.
    private val labels = Vector<String>()
    private var tfLite: Interpreter? = null

    override fun detectOnImage(bitmap: Bitmap): List<DetectionResult> {
        val byteBuffer = convertBitmapToByteBuffer(bitmap)
        val detections: List<DetectionResult>
        var startTime = SystemClock.uptimeMillis()
        detections = if (isTiny) {
            getDetectionsForTiny(byteBuffer, bitmap)
        } else {
            getDetectionsForFull(byteBuffer, bitmap)
        }
        Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime))
        return nms(detections)
    }

    //non maximum suppression
    protected fun nms(list: ArrayList<DetectionResult>): List<DetectionResult> {
        val nmsList: MutableList<DetectionResult> = mutableListOf<DetectionResult>()
        for (k in labels.indices) {
            //1.find max confidence per class
            val pq: PriorityQueue<DetectionResult> = PriorityQueue<DetectionResult>(
                    50,
                    Comparator<DetectionResult> { lhs, rhs -> // Intentionally reversed to put high confidence at the head of the queue.
                        rhs.confidence.compareTo(lhs.confidence)
                    })
            for (i in list.indices) {
                if (list[i].detectedClass === k) {
                    pq.add(list[i])
                }
            }

            //2.do non maximum suppression
            while (pq.size > 0) {
                //insert detection with max confidence
                val a: Array<DetectionResult?> = arrayOfNulls<DetectionResult>(pq.size)
                val detections: Array<DetectionResult> = pq.toArray(a)
                val max: DetectionResult = detections[0]
                nmsList.add(max)
                pq.clear()
                for (j in 1 until detections.size) {
                    val detection: DetectionResult = detections[j]
                    val b: RectF = detection.locationRectF
                    if (boxIou(max.locationRectF, b) < mNmsThresh) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsList
    }

    protected var mNmsThresh = 0.6f
    protected fun boxIou(a: RectF, b: RectF): Float {
        return boxIntersection(a, b) / boxUnion(a, b)
    }

    protected fun boxIntersection(a: RectF, b: RectF): Float {
        val w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left)
        val h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top)
        return if (w < 0 || h < 0) 0.0f else w * h
    }

    protected fun boxUnion(a: RectF, b: RectF): Float {
        val i = boxIntersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    protected fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float {
        val l1 = x1 - w1 / 2
        val l2 = x2 - w2 / 2
        val left = if (l1 > l2) l1 else l2
        val r1 = x1 + w1 / 2
        val r2 = x2 + w2 / 2
        val right = if (r1 < r2) r1 else r2
        return right - left
    }

    /**
     * Writes Image data into a `ByteBuffer`.
     */
    protected fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val numBytesPerChannel: Int = if (this.isModelQuantized)
            1 // Quantized
        else
            4 // Floating point

        val byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE * numBytesPerChannel)
        byteBuffer.order(ByteOrder.nativeOrder())
//        val matrix: Matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
//                inputSize, inputSize, false)
//        val bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
//        val canvas = Canvas(bitmap)
//        canvas.drawBitmap(bitmapRaw, matrix, null)
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val `val` = intValues[pixel++]
                byteBuffer.putFloat((`val` shr 16 and 0xFF) / 255.0f)
                byteBuffer.putFloat((`val` shr 8 and 0xFF) / 255.0f)
                byteBuffer.putFloat((`val` and 0xFF) / 255.0f)
            }
        }
        return byteBuffer
    }

    /**
     * For yolov4-tiny, the situation would be a little different from the yolov4, it only has two
     * output. Both has three dimenstion. The first one is a tensor with dimension [1, 2535,4], containing all the bounding boxes.
     * The second one is a tensor with dimension [1, 2535, class_num], containing all the classes score.
     * @param byteBuffer input ByteBuffer, which contains the image information
     * @param bitmap pixel disenty used to resize the output images
     * @return an array list containing the DetectionResults
     */
    private fun getDetectionsForFull(byteBuffer: ByteBuffer, bitmap: Bitmap): ArrayList<DetectionResult> {
        val detections: ArrayList<DetectionResult> = ArrayList()
        val outputMap: MutableMap<Int, Any> = HashMap()
        outputMap[0] = Array(1) { Array(OUTPUT_WIDTH_FULL[0]) { FloatArray(4) } }
        outputMap[1] = Array(1) { Array(OUTPUT_WIDTH_FULL[1]) { FloatArray(labels.size) } }
        val inputArray = arrayOf<Any>(byteBuffer)
        tfLite!!.runForMultipleInputsOutputs(inputArray, outputMap)
        val gridWidth = OUTPUT_WIDTH_FULL[0]
        val bboxes = outputMap[0] as Array<Array<FloatArray>>?
        val out_score = outputMap[1] as Array<Array<FloatArray>>?
        for (i in 0 until gridWidth) {
            var maxClass = 0f
            var detectedClass = -1
            val classes = FloatArray(labels.size)
            for (c in labels.indices) {
                classes[c] = out_score!![0][i][c]
            }
            for (c in labels.indices) {
                if (classes[c] > maxClass) {
                    detectedClass = c
                    maxClass = classes[c]
                }
            }
            val score = maxClass
            if (score > objThresh) {
                val xPos = bboxes!![0][i][0]
                val yPos = bboxes[0][i][1]
                val w = bboxes[0][i][2]
                val h = bboxes[0][i][3]
                val rectMap: Map<String, Float> = mapOf("left" to max(0f, xPos - w / 2),
                "top" to max(0f, yPos - h / 2),
                "right" to min((bitmap.width - 1).toFloat(), xPos + w / 2),
                "bottom" to min((bitmap.height - 1).toFloat(), yPos + h / 2))
//                val rectF = RectF(
//                        max(0f, xPos - w / 2),
//                        max(0f, yPos - h / 2),
//                        min((bitmap.width - 1).toFloat(), xPos + w / 2),
//                        min((bitmap.height - 1).toFloat(), yPos + h / 2))
                detections.add(DetectionResult("" + i, labels[detectedClass], score, rectMap, detectedClass))
            }
        }
        return detections
    }

    private fun getDetectionsForTiny(byteBuffer: ByteBuffer, bitmap: Bitmap): ArrayList<DetectionResult> {
        val detections: ArrayList<DetectionResult> = ArrayList()
        val outputMap: MutableMap<Int, Any> = HashMap()
        outputMap[0] = Array(1) { Array(OUTPUT_WIDTH_TINY[0]) { FloatArray(4) } }
        outputMap[1] = Array(1) { Array(OUTPUT_WIDTH_TINY[1]) { FloatArray(labels.size) } }
        val inputArray = arrayOf<Any>(byteBuffer)
        tfLite!!.runForMultipleInputsOutputs(inputArray, outputMap)
        val gridWidth = OUTPUT_WIDTH_TINY[0]
        val bboxes = outputMap[0] as Array<Array<FloatArray>>?
        val out_score = outputMap[1] as Array<Array<FloatArray>>?
        for (i in 0 until gridWidth) {
            var maxClass = 0f
            var detectedClass = -1
            val classes = FloatArray(labels.size)
            for (c in labels.indices) {
                classes[c] = out_score!![0][i][c]
            }
            for (c in labels.indices) {
                if (classes[c] > maxClass) {
                    detectedClass = c
                    maxClass = classes[c]
                }
            }
            val score = maxClass
            if (score > objThresh) {
                val xPos = bboxes!![0][i][0]
                val yPos = bboxes[0][i][1]
                val w = bboxes[0][i][2]
                val h = bboxes[0][i][3]
                val rectMap: Map<String, Float> = mapOf("left" to max(0f, xPos - w / 2),
                        "top" to max(0f, yPos - h / 2),
                        "right" to min((bitmap.width - 1).toFloat(), xPos + w / 2),
                        "bottom" to min((bitmap.height - 1).toFloat(), yPos + h / 2))
//                val rectF = RectF(
//                        max(0f, xPos - w / 2),
//                        max(0f, yPos - h / 2),
//                        min((bitmap.width - 1).toFloat(), xPos + w / 2),
//                        min((bitmap.height - 1).toFloat(), yPos + h / 2))
                detections.add(DetectionResult("" + i, labels[detectedClass], score, rectMap, detectedClass))
            }
        }
        return detections
    }


    companion object {
        /**
         * Initializes a native TensorFlow session for classifying images.
         *
         * @param assetManager  The asset manager to be used to load assets.
         * @param modelFilename The filepath of the model GraphDef protocol buffer.
         * @param labelFilename The filepath of label file for classes.
         * @param isQuantized   Boolean representing model is quantized or not
         * @param isTiny   Boolean representing model is Yolo tiny or not
         * @param isGPU   If true, uses GPU for inference
         * @param isNNAPI   If true uses the Neural Network API
         * @param isModelAnAsset   If true, loads the model file from the assets/ folder, else directly from the model path
         */
        @Throws(IOException::class)
        fun create(
                assetManager: AssetManager,
                modelFilename: String,
                labelFilename: String,
                isQuantized: Boolean,
                isTiny: Boolean = true,
                isGPU: Boolean = true,
                isNNAPI: Boolean = false,
                isModelAnAsset: Boolean = false): ObjectDetector {
            val d = YoloV4()

            // Load labels
            val labelsInput = assetManager.open(labelFilename)
            BufferedReader(InputStreamReader(labelsInput)).forEachLine { d.labels.add(it) }

            // Initialize interpreter and load model
            try {
                val options = Interpreter.Options()
                options.setNumThreads(NUM_THREADS)
                if (isNNAPI) {
                    var nnApiDelegate: NnApiDelegate? = null
                    // Initialize interpreter with NNAPI delegate for Android Pie or above
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                        nnApiDelegate = NnApiDelegate()
                        options.addDelegate(nnApiDelegate)
                        options.setNumThreads(NUM_THREADS)
                        options.setUseNNAPI(false)
                        options.setAllowFp16PrecisionForFp32(true)
                        options.setAllowBufferHandleOutput(true)
                        options.setUseNNAPI(true)
                    }
                }
                if (isGPU) {
                    val gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                }
                d.tfLite = Interpreter(Utils.loadModelFile(assetManager, modelFilename, isModelAnAsset), options)
            } catch (e: Exception) {
                throw RuntimeException(e)
            }
            d.isModelQuantized = isQuantized
            d.isTiny = isTiny
            return d
        }

        // Float model
        private const val IMAGE_MEAN = 0f
        private const val IMAGE_STD = 255.0f

        //config yolov4
        private const val INPUT_SIZE = 416
        private val OUTPUT_WIDTH = intArrayOf(52, 26, 13)
        private val MASKS = arrayOf(intArrayOf(0, 1, 2), intArrayOf(3, 4, 5), intArrayOf(6, 7, 8))
        private val ANCHORS = intArrayOf(
                12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
        )
        private val XYSCALE = floatArrayOf(1.2f, 1.1f, 1.05f)
        private const val NUM_BOXES_PER_BLOCK = 3

        // Number of threads in the app
        private const val NUM_THREADS = 4

        // config yolov4 tiny
        private val OUTPUT_WIDTH_TINY = intArrayOf(2535, 2535)
        private val OUTPUT_WIDTH_FULL = intArrayOf(10647, 10647)
        private val MASKS_TINY = arrayOf(intArrayOf(3, 4, 5), intArrayOf(1, 2, 3))
        private val ANCHORS_TINY = intArrayOf(
                23, 27, 37, 58, 81, 82, 81, 82, 135, 169, 344, 319)
        private val XYSCALE_TINY = floatArrayOf(1.05f, 1.05f)
        protected const val BATCH_SIZE = 1
        protected const val PIXEL_SIZE = 3
    }
}