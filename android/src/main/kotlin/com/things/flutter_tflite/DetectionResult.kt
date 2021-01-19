package com.things.flutter_tflite

import android.graphics.RectF
import kotlinx.serialization.*

/**
 * An immutable result returned by an ObjectDetector describing what was detected.
 * @property id A unique identifier for what has been recognized. Specific to the class, not the instance of the object.
 * @property title Display name for the detection.
 * @property confidence A sortable score for how good the detection is relative to others. Higher should be better.
 * @property location Optional location within the source image for the location of the recognized object.
 * @property detectedClass
 */
@Serializable
data class DetectionResult(val id: String?, val title: String?, val confidence: Float, val location: Map<String, Float>, var detectedClass: Int) {
    @Transient var locationRectF: RectF = RectF(location["left"]!!, location["top"]!!, location["right"]!!, location["bottom"]!!)

    override fun toString(): String {
        var resultString = ""
        if (id != null) {
            resultString += "[$id] "
        }
        if (title != null) {
            resultString += "$title "
        }
        resultString += "${"%.1f".format(confidence * 100.0f)}% "
        resultString += "$location "
        return resultString.trim { it <= ' ' }
    }
}
