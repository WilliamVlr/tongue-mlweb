from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    #Create a webcam object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int (cap.get(3))
    frame_height = int (cap.get(4))

    model = YOLO('best.pt')
    classNames = ['Geographically', 'Geographically Normal', 'Geographically Retak', 'Normal', 'Retak'] #Harus sesuai urutan folder train datasetnya
    while True:
        success, img = cap.read()
        if not success:
            print("Not successful")
            break
        results = model(img, stream=True)
        for r in results:
            if r is not None:
                prob = r.probs
                idx = prob.top1
                conf = prob.top1conf
                if idx < len(classNames):
                    class_name = classNames[idx]
                    label = f'{class_name} {conf.item():.4f}'
                    cv2.putText(img, label, (10, 30), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    # Handle the case where the index is out of range
                    print(f"Warning: Predicted class index {idx} is out of range")
                
        # print(results)
        yield img
    cap.release()
    cv2.destroyAllWindows()