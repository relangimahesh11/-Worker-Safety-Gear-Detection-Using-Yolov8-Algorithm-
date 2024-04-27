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
![train_batch2](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/4c03f7ca-370e-4871-bc87-c0271ee8840d)
![train_batch1](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/26ac8c9f-e972-4dec-811c-6a479ef0f14d)
![train_batch0](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/dcf49f9c-b2fc-41e4-bee5-d8d34429f48f)
![results](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/a61fd24c-0388-4486-88c6-a01bb65498cb)
![R_curve](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/0012d6aa-07be-4018-af74-2ebaeb944ffd)
![PR_curve](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/09075de8-e3ca-4b7e-8336-4e7585e18f17)
![P_curve](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/8d2f3d2d-cec8-42bb-899f-babbdfa8bc9d)
![labels_correlogram](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/866a94e3-63dc-4e2e-92fa-2ebadbf43b7a)
![labels](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/d7ba10d5-f4df-4d7c-b6c7-523a852e0511)
![F1_curve](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/8e529673-c0e4-4efc-82af-b53393e5dc12)
![confusion_matrix](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/dbdaed37-891c-4693-8083-1f78998e2abd)
![train_batch14672](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/46ae6b85-28fe-4b44-a1ed-2f6d1aae32aa)
![train_batch14671](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/72a0d5a8-c922-4cbe-b6bf-ff63c75301c2)
![train_batch14670](https://github.com/relangimahesh11/-Worker-Safety-Gear-Detection-Using-Yolov8-Algorithm-/assets/166863593/1503a2cc-07ef-47d6-bcb3-f87c6b432fbc)
