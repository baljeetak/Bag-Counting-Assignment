import cv2
from ultralytics import YOLO

# 1. Load the fastest model
model = YOLO('yolov8n.pt') 

video_path = "Problem Statement Scenario1.mp4" 
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('bag_counter_method2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# --- CONFIGURATION ---
# Line at 60% (Right in the middle of their walking path)
line_x = int(width * 0.5) 
bag_counter = 0
counted_ids = set()
track_history = {} 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. THE SECRET: TRACK ONLY PEOPLE (classes=[0])
    # By only tracking people, we get a perfect proxy count of the bags they are carrying.
    results = model.track(frame, persist=True, classes=[0], conf=0.25)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)

            # Draw UI around the Worker (Carrier)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Carrier ID:{track_id}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 3. RIGHT-TO-LEFT COUNTING LOGIC (Warehouse to Truck)
            if track_id in track_history:
                prev_cx = track_history[track_id]
                
                # If they were on the Right, and moved to the Left, count the bag!
                if prev_cx >= line_x and cx < line_x:
                    if track_id not in counted_ids:
                        bag_counter += 1
                        counted_ids.add(track_id)
            
            track_history[track_id] = cx

    # Visuals
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 3)
    cv2.rectangle(frame, (20, 20), (350, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"BAGS LOADED: {bag_counter}", (40, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    out.write(frame)
    cv2.imshow("Bag Counting System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Final Count: {bag_counter}")