import sys
sys.path.insert(0, "D:/Shehsaath/Original_new/YOLOv8-multi-task/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    # Load a model
    model = YOLO('D:/Shehsaath/Original_new/YOLOv8-multi-task/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-s.yaml', task='multi')  # build a new model from YAML
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    model.train(data='D:/Shehsaath/Original_new/YOLOv8-multi-task/ultralytics/datasets/bdd-multi2.yaml', batch=8, epochs=100, imgsz=(640,640), device=[0], name='yolopm', val=True, task='multi', classes=[0,1,2,3,4,5,6,7,8,9,10,11],single_cls=False)
# classes = [2,3,4,5,6,10,11]

# import sys
# # Make sure this path is correct for your system
# sys.path.insert(0, "./")
# from ultralytics import YOLO

# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.freeze_support() 
#     # Load a model
#     # MODIFIED: Point to your new 2-task model architecture
#     model = YOLO('D:/Shehsaath/Original_new/YOLOv8-multi-task/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-s.yaml', task='multi')  # build a new model from YAML

#     # Train the model
#     # MODIFIED: Point to your new 2-task dataset configuration
#     model.train(data='D:/Shehsaath/Original_new/YOLOv8-multi-task/ultralytics/datasets/bdd-multi2.yaml',
#                 batch=8,
#                 epochs=30,
#                 imgsz=(640,640),
#                 device=[0],
#                 name='det_lane_experiment', # You can change the experiment name
#                 val=True,
#                 task='multi',
#                 combine_class=[2,3,4,5,6,7,8], # Combine classes for detection
#                 single_cls=False)
#                 # amp= False