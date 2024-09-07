import math
from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

classNames = [
    "gun",
    "Knife"
]

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# model = YOLO("yolov8n.pt")
model = YOLO("YOLO_ThreatDetection_v1.0.pt")

def gen_frames():
    while True:
        success, frame = camera.read()
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                cls = int(box.cls[0])
                classname = classNames[cls] if cls < len(classNames) else "Unknown"
                print("Cls -->", box.cls)
                print("Class name -->", classname)

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(
                    frame, classname, org, font, fontScale, color, thickness
                )

        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            ) 


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)