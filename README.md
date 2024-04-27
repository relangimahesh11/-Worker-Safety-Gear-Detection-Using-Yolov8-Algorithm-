*This goal of the project is to detecting deviations and localizing safety gear worn by workers, such as helmets, vests, mask.
*The primary goal is to ensure that workers are properly equipped with necessary safety gear.
*Safety in Industries is paramount for protecting workers from accidents with heavy machinery and hazardous materials, ensuring compliance with regulations, and maintaining a culture of safety.
*Traditional safety monitoring methods in industrial environments are time-consuming and prone to human error.
*This is addressed by the project, “Worker safety gear detection using YOLOv8 algorithm” addresses this by using Deep Learning Capability
to detect safety violations. 
*Through our research currently considering YOLOv8 algorithm to detect safety care of industrial worker. YOLOv8 is mostly used for object
detection in machine learning and deep learning.


import cv2
from ultralytics import YOLO

best_model = "/home/sanjay/Music/train6/weights/best.pt"
model = YOLO(best_model)
cap = cv2.VideoCapture('0')  # Change the video source as needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        continue

    try:
        results = model.predict(source=frame, save=False)
        for result in results:
            boxes = result.boxes.data.cpu().numpy()  # Get boxes as numpy array
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box[:8]  # Unpack box data
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{model.names[int(cls)]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        cv2.imshow('Object Detection', frame)
        except Exception as e:
        print(f"Error processing camera frame: {e}")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        cap.release()
![WhatsApp Image 2024-04-26 at 12 52 26_33dc6819](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/96b977a1-441e-4b76-a1ef-cda296255649)
![WhatsApp Image 2024-04-26 at 12 51 01_aafdcf43](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/fe3f7508-36c8-451f-85bb-995cb912b337)
