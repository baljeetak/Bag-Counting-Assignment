import cv2
from ultralytics import YOLO

# 1. Load the model
model = YOLO('yolov8n.pt') 

video_path = "Problem Statement Scenario1.mp4" 
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('bag_counter_method1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# --- CONFIGURATION ---
line_x = int(width * 0.5) 
bag_counter = 0
counted_ids = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. TRACKING WITH AGNOSTIC NMS
    results = model.track(frame, persist=True, conf=0.08, iou=0.5, agnostic_nms=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            
            # STRICTLY IGNORE HUMANS
            if cls == 0:
                continue 
            
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cx = int((x1 + x2) / 2)
            area = w * h

            # 3. RELAXED BAG FILTERING LOGIC
            is_bag = False
            
            # Condition A: Standard Bag classes
            if cls in [24, 26, 28]: 
                is_bag = True
                
            elif area > 500 and h < (height * 0.65): 
                is_bag = True

            if is_bag:
                # Draw Bag UI (Blue Box)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
                # 4. Counting Logic 
                if cx > line_x and track_id not in counted_ids:
                    bag_counter += 1
                    counted_ids.add(track_id)

    # Visuals
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)
    cv2.rectangle(frame, (20, 20), (350, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"BAG COUNT: {bag_counter}", (40, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    out.write(frame)
    cv2.imshow("Bag Counting System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Final Count: {bag_counter}")
