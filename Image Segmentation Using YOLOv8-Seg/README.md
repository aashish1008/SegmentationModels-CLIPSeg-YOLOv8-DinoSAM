# YOLOv8-Seg: Real-Time Image Segmentation
YOLOv8-Seg enhances the YOLO architecture to perform pixel-level segmentation, providing segmentation masks alongside bounding box predictions. It is suitable for real-time applications due to its efficient performance.

###### Features
- **Variants:** Includes `yolov8n-seg`, `yolov8s-seg`, `yolov8m-seg`, `yolov8l-seg`, and `yolov8x-seg`.
- **Customizable:** High accuracy when trained on custom datasets.
- **Versatile:** Suitable for both detection and segmentation tasks in dynamic environments.
- **Performance:** Achieves approximately 30 FPS with Nvidia GTX 1650 GPU, making it practical for real-time applications.

- ## Installation
To set up YOLOv8-Seg, follow these steps:

1. **Clone the repository:**
   ``` bash
   git clone https://github.com/aashish1008/SegmentationModels-CLIPSeg-YOLOv8-DinoSAM.git
   cd SegmentationModels-CLIPSeg-YOLOv8-DinoSAM/YOLOv8-Seg
2. **Install dependencies:**
  It's recommended to use a virtual environment. You can set up a virtual environment using venv or conda.
  For `conda`:
    ``` bash
    conda create --name yolov8_seg python=3.10
    conda activate yolov8_seg
    pip install -r requirements.txt 

## Usage
1. Prepare your dataset: Ensure you have your images and annotations ready for training and testing.
2. Train your dataset in Yolov8 Segmentation Model.
3. Download the model which can be your `best.pt` or `last.pt`
4. Setup your trained model in python file
5. Run your python file:
   ``` bash
   python yoloseg_sheep.py

## Results
After running the script, the segmentation results will be saved in the output directory. You can visualize the results to see how well the model performed on the given images or videos.

## Contributing
Contributions and feedback are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.
