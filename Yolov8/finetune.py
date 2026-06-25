import sys
# Make sure to adjust this path if necessary
sys.path.insert(0, "D:/Shehsaath/Original_new/YOLOv8-multi-task")

from ultralytics import YOLO

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    # --- Fine-Tuning Script ---

    # 1. Define the paths to your custom model and dataset YAML files
    # Using the 2-task fusion model we designed
    model_yaml_path = 'D:/Shehsaath/Original_new/YOLOv8-multi-task/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-s.yaml'
    dataset_yaml_path = 'D:/Shehsaath/Original_new/YOLOv8-multi-task/ultralytics/datasets/bdd-multi2.yaml' # Your 2-task dataset yaml
    pretrained_weights = "D:/Shehsaath/Yolov8_github/YOLOv8-multi-task/runs/multi/yolopm29/weights/best.pt" # Standard pre-trained weights

    # 2. Build the custom multi-task model from your YAML file
    print(f"Building custom model structure from {model_yaml_path}...")
    model = YOLO(model_yaml_path, task='multi')

    # 3. Load the pre-trained weights into the matching layers (the backbone)
    print(f"Loading pre-trained backbone weights from {pretrained_weights}...")
    model.load(pretrained_weights)

    # 4. Start fine-tuning the model on your dataset with a lower learning rate
    print("Starting fine-tuning...")
    model.train(data=dataset_yaml_path,
                batch=8,
                epochs=3,
                imgsz=240,
                device=[0],
                name='yolov8_finetuned_fusion_2task',
                val=True,
                classes=[2, 3, 4, 7, 8, 9, 10, 11],
                task='multi',
                lr0=0.001) # Lower learning rate for fine-tuning

    print("Fine-tuning complete.")