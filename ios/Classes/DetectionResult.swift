/**
 * An immutable result returned by an ObjectDetector describing what was detected.  
 */
public class DetectionResult: CustomStringConvertible {

    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of the object.
     */
    var id: String;

    /**
     * Display name for the recognition.
     */
    var title: String;

    /**
     * A sortable score for how good the detection is relative to others. Higher should be better.
     */
    var confidence: Float;

    /**
     * Optional location within the source image for the location of the recognized object.
     */
    var location: CGRect;

    /**
     * Display name for the recognition.
     */
    var detectedClass: Int;

    init?(id:String,title:String,confidence: Float,location:CGRect, detectedClass:Int) {
        self.id = id;
        self.title = title;
        self.confidence = confidence;
        self.location = location;
        self.detectedClass = detectedClass;
    }

    public var description: String{
        var resultString : String = "{"
        if (id != nil) {
            resultString = resultString + "\"id\": " + id + ","
        }

        if (title != nil) {
            resultString = resultString + "\"detectedClass\": \"" + title + "\","
        }

        resultString = resultString + "\"confidenceInClass\": " +  String(confidence) + ","
        resultString = resultString + "\"rect\": { \"t\": " + location.minX.description + ", \"b\": " + location.height.description + ", \"l\": " + location.minY.description + ", \"r\": " + location.width.description + "}"
        resultString = resultString + "}"
        return resultString
    }

}