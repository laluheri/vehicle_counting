import cv2
from ultralytics import YOLOv10
import numpy as np
from collections import defaultdict
import time
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="yolov10m.pt", help="YOLOv10 model path")
parser.add_argument("--video", type=str, default="inference/highway.mp4", help="Path to input video or webcam index (0)")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence Threshold for detection")
args = parser.parse_args()

# Function to display FPS on the frame
def show_fps(frame, fps):
    x, y, w, h = 10, 10, 350, 50
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)  # Draw a rectangle for the FPS text background
    cv2.putText(frame, "FPS: " + str(fps), (20, 52), cv2.FONT_HERSHEY_PLAIN, 3.5, (0, 255, 0), 3)  # Add FPS text

# Function to display Counter on the frame
def show_counter(frame, title, vehicle_count, x_init):
    vehicle_names = {
        1: "bicycle",
        2: "car",
        3: "motorcycle",  
        5: "bus", 
        7: "truck" 
    }    

    # Show Counters
    y_init = 100
    gap = 30    

    cv2.rectangle(frame, (x_init - 5, y_init - 35), (x_init + 200, 265), (0, 255, 0), -1)    
    
    cv2.putText(frame, title, (x_init, y_init - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for vehicle_id, count in vehicle_count.items():
        y_init += gap

        vehicle_name = vehicle_names[vehicle_id]
        vehicle_count = "%.3i" % (count)
        cv2.putText(frame, vehicle_name, (x_init, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)            
        cv2.putText(frame, vehicle_count, (x_init + 140, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

if __name__ == '__main__':
    # Set up video capture
    video_input = args.video
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)  # Open webcam if video_input is a digit
    else:
        cap = cv2.VideoCapture(video_input)  # Open video file    

    conf_thres = args.conf  # Confidence threshold for detection

    model = YOLOv10(args.model)  # Load YOLOv10 model

    track_history = defaultdict(lambda: [])  # Track history of detected objects
    start_time = 0  # Initialize start time for FPS calculation

    # Count Line
    entered_vehicle_ids = []
    exited_vehicle_ids = []    

    vehicle_class_ids = [1, 2, 3, 5, 7]

    vehicle_entry_count = {
        1: 0,  # bicycle
        2: 0,  # car
        3: 0,  # motorcycle
        5: 0,  # bus
        7: 0   # truck
    }
    vehicle_exit_count = {
        1: 0,  # bicycle
        2: 0,  # car
        3: 0,  # motorcycle
        5: 0,  # bus
        7: 0   # truck
    }

    entry_line = {
        'x1' : 160, 
        'y1' : 558,  
        'x2' : 708,  
        'y2' : 558,          
    }
    exit_line = {
        'x1' : 1155, 
        'y1' : 558,  
        'x2' : 1718,  
        'y2' : 558,          
    }
    offset = 80

    # Variabel total kendaraan
    total_enter = 0
    total_exit = 0

    while cap.isOpened():
        success, frame = cap.read()  # Read a frame from the video
        annotated_frame = frame

        if success:
            # Draw Count Line
            cv2.line(frame, (entry_line['x1'], entry_line['y1']), (exit_line['x2'], exit_line['y2']), (0, 127, 255), 3)

            # Perform object tracking using YOLOv10
            results = model.track(frame, classes=vehicle_class_ids, persist=True, tracker="bytetrack.yaml", conf=conf_thres)

            boxes = results[0].boxes.xywh.cpu()  # Get bounding boxes

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()  # Get track IDs
                class_ids = results[0].boxes.cls.int().cpu().tolist()  # Get class IDs

                # Plot the results on the frame
                annotated_frame = results[0].plot()

                # Draw tracking lines for each object
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # Append center point of the bounding box
                    if len(track) > 30:  # Retain track history for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(255, 0, 0),
                        thickness=3,
                    )

                    # Counter               
                    center_x = int(x)               
                    center_y = int(y + (h / 2))                                        

                    if((center_x in range(entry_line['x1'], entry_line['x2'])) and (center_y in range(entry_line['y1'], entry_line['y1'] + offset)) ):
                        if (int(track_id) not in entered_vehicle_ids):  
                            vehicle_entry_count[class_id] += 1
                            entered_vehicle_ids.append(int(track_id))

                    if((center_x in range(exit_line['x1'], exit_line['x2'])) and (center_y in range(exit_line['y1'] - offset, exit_line['y1'])) ):                        
                        if(int(track_id) not in exited_vehicle_ids):                    
                            vehicle_exit_count[class_id] += 1                  
                            exited_vehicle_ids.append(int(track_id)) 

            # Hitung total kendaraan yang masuk dan keluar
            total_enter = sum(vehicle_entry_count.values())
            total_exit = sum(vehicle_exit_count.values())

            # Show Counter on the frame
            show_counter(annotated_frame, "Vehicle Enter", vehicle_entry_count, 10)
            show_counter(annotated_frame, "Vehicle Exit", vehicle_exit_count, 1710)

            # Tampilkan total kendaraan yang masuk dan keluar
            cv2.putText(annotated_frame, f"Total Entered: {total_enter}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Total Exited: {total_exit}", (1710, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            start_time = end_time

            # Show FPS on the frame
            fps = float("{:.2f}".format(fps))
            show_fps(annotated_frame, fps)

            # Display the annotated frame
            imS = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow('YOLOv10 Vehicle Counter', imS)                                    

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
