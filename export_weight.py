import os 
from ultralytics import YOLO

def generate_onnx_model():
    print("=" * 60)
    print("MOSQAI MODEL COMPILATION MATRIX INITIALIZATION")
    print("=" * 60)

    target_dir = "models"
    os.makedirs(target_dir, exist_ok=True)
    output_onnx_path = os.path.join(target_dir, "yolov8n_mosquito.onnx")
    
    base_pytorch_weights = "yolov8n.pt" 
    
    print(f"[STEP 1]: Loading base weights layer from: {base_pytorch_weights}...")
    model = YOLO(base_pytorch_weights)
    
    print("\n[STEP 2]: Executing ONNX mathematical layer serialization...")
    print("Configuring parameters: ImgSize=640, Opset=12, Simplify=True...")

    model.export(format="onnx", imgsz=640, opset=12, simplify=True)
    
    default_export_output = base_pytorch_weights.replace(".pt", ".onnx")
    if os.path.exists(default_export_output) and default_export_output != output_onnx_path:
        print(f"\n[STEP 3]: Moving compiled asset to target destination folder...")
        os.replace(default_export_output, output_onnx_path)
        
    print("\n" + "=" * 60)
    print(f"[SUCCESS]: Asset generated at: {output_onnx_path}")
    print("=" * 60)

if __name__ == "__main__":
    generate_onnx_model()
