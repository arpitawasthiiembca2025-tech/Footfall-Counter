"""
footfall_yolov8_realtime_vertical.py
Real-time footfall counter using YOLOv8 + Centroid tracking (Vertical counting line).

Requirements:
    pip install ultralytics opencv-python numpy
Usage:
    python footfall_yolov8_realtime_vertical.py --source 0
    python footfall_yolov8_realtime_vertical.py --line_x 400
"""

import argparse
import time
from collections import OrderedDict, deque
import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------
# Centroid Tracker
# --------------------------
class TrackableObject:
    def __init__(self, object_id, centroid, bbox):
        self.object_id = object_id
        self.centroids = deque(maxlen=30)
        self.centroids.append(centroid)
        self.bbox = bbox
        self.counted = False


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=60):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        obj = TrackableObject(self.next_object_id, centroid, bbox)
        self.objects[self.next_object_id] = obj
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            to_deregister = []
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    to_deregister.append(object_id)
            for oid in to_deregister:
                self.deregister(oid)
            return self.objects

        input_centroids = []
        for (startX, startY, endX, endY) in rects:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for i, (cX, cY) in enumerate(input_centroids):
                self.register((cX, cY), rects[i])
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [obj.centroids[-1] for obj in self.objects.values()]

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype="float")
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id].centroids.append(input_centroids[col])
            self.objects[object_id].bbox = rects[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col], rects[col])

        return self.objects


# --------------------------
# Main Function
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0", help="0 (webcam) or path to video")
    p.add_argument("--line_x", type=int, default=None, help="x position of vertical counting line")
    p.add_argument("--display", action="store_true", help="Show video window")
    p.add_argument("--save_out", type=str, default=None, help="Save processed video to file")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model to use")
    return p.parse_args()


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25.0

    line_x = args.line_x if args.line_x else width // 2

    writer = None
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_out, fourcc, fps, (width, height))

    tracker = CentroidTracker(max_disappeared=30, max_distance=80)
    total_in, total_out = 0, 0
    frame_count = 0
    start_time = time.time()

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            cls = int(box.cls.cpu().numpy().item())
            if cls != 0:
                continue
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            startX, startY, endX, endY = xyxy
            boxes.append((startX, startY, endX, endY))

        objects = tracker.update(boxes)

        for object_id, obj in objects.items():
            if len(obj.centroids) < 2:
                continue
            prev_x = obj.centroids[-2][0]
            curr_x = obj.centroids[-1][0]

            if not obj.counted:
                if prev_x < line_x and curr_x >= line_x:
                    total_in += 1
                    obj.counted = True
                elif prev_x > line_x and curr_x <= line_x:
                    total_out += 1
                    obj.counted = True

        vis = frame.copy()
        cv2.line(vis, (line_x, 0), (line_x, height), (0, 255, 255), 2)
        cv2.putText(vis, f"IN: {total_in}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        cv2.putText(vis, f"OUT: {total_out}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)

        for object_id, obj in objects.items():
            (x1, y1, x2, y2) = obj.bbox
            cX, cY = obj.centroids[-1]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"ID {object_id}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.circle(vis, (cX, cY), 4, (0, 0, 255), -1)

        fps_text = f"FPS: {frame_count / (time.time() - start_time):.2f}"
        cv2.putText(vis, fps_text, (width - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        if args.display or True:
            cv2.imshow("Footfall Counter (Vertical Line)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer:
            writer.write(vis)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"[RESULT] Total IN: {total_in} | Total OUT: {total_out}")
    print("[INFO] Finished.")


if __name__ == "__main__":
    main()
