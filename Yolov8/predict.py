import sys
sys.path.insert(0, "/home/jiayuan/ultralytics-main/ultralytics")

from ultralytics import YOLO

if __name__ == '__main__':
    number = 3 #input how many tasks in your work
    model = YOLO(r"D:\Shehsaath\Final trained weights collection\bdd_weights\C3TR weight\best.pt")  # Validate the model
    model.predict(source=r"D:\Shehsaath\New annotations Oct\images to include in paper\bdd dataset\Original images", imgsz=(384,672), device=[0],name='v4_daytime', save=True, conf=0.25, iou=0.45, show_labels=False) #, classes=[4, 6, 7, 8, 9])
