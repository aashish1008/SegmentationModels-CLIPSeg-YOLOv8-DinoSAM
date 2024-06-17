# SegmentationModels-CLIPSeg-YOLOv8-DinoSAM
Exploring Zero-Shot Model Capabilities in Image Segmentation

Watch the demo on YouTube
https://youtu.be/28ZZTCqmSQU

I have also written a blog about this model. If you're interested, you can read it here:
https://medium.com/@aashishstha34/zero-shot-image-segmentation-from-clipseg-to-yolov8-seg-vs-grounding-dino-with-sam-83706cb601a6
## Models Explored

### CLIPSeg
CLIPSeg utilizes text and vision embeddings to generate segmentation masks based on textual prompts. It bridges the gap between natural language processing and computer vision, enabling zero-shot image segmentation.
- Text transformer for processing textual descriptions.
- Visual transformer for handling visual inputs.
- Seamless integration of text and vision embeddings for segmentation.
- Optimized for static image segmentation while less effective for dynamic video frames.

### YOLOv8-Seg
YOLOv8-Segmentation Model enhances YOLO architecture to perform pixel-level segmentation. It provides segmentation masks alongside bounding box predictions, suitable for real-time applications.
- Variants include yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, and yolov8x-seg.
- Customizable and adaptable with high accuracy when trained on custom datasets.
- Suitable for both detection and segmentation tasks in dynamic environments.
- Achieves approximately 30 FPS with moderate GPU requirements, making it practical for real-time applications.

### Grounding DINO with SAM
Grounding DINO with SAM integrates Grounding DINO (open-shot object detector) with SAM (Segment Anything Model) for enhanced zero-shot segmentation. It detects objects based on textual cues and generates precise segmentation masks.
- Text-guided detection for identifying objects based on arbitrary text inputs.
- SAM facilitates detailed segmentation mask generation for detected objects.
- High computational demands make real-time implementation challenging specially for video frames.

## Repository Structure
- `CLIPSeg/`: Implementation and experiments with CLIPSeg model.
- `Image Segmentation Using YOLOv8-Seg/`: Variants and performance evaluations of YOLOv8-Seg.
- `GroundingDINO with SAM/`: Integration and experiments with Grounding DINO and SAM.

## Getting Started
To explore these models and reproduce experiments:

1. Clone the repository:
   ``` bash
   git clone https://github.com/aashish1008/SegmentationModels-CLIPSeg-YOLOv8-DinoSAM.git
   cd SegmentationModels-CLIPSeg-YOLOv8-DinoSAM
2. Follow instructions in each model directory (`CLIPSeg/`, `Image Segmentation Using YOLOv8-Seg/`, `GroundingDINO with SAM/`) to set up dependencies and run experiments.
3. Refer to specific README files in each directory for detailed setup and usage instructions.

## Contributing
Contributions and feedback are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

