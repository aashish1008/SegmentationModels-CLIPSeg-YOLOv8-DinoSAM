# Grounding DINO with SAM: Zero-Shot Image Segmentation
Grounding DINO with SAM integrates Grounding DINO (open-set object detector) with SAM (Segment Anything Model) to enhance zero-shot segmentation capabilities. It detects objects based on textual cues and generates precise segmentation masks.

#### Features
- **Text-Guided Detection:** Identifies objects based on textual descriptions.
- **Accurate Segmentation:** Generates precise segmentation masks for detected objects.
- **Zero-Shot Learning:** No need for extensive annotated training data.
#### Limitations
- High computational demands make real-time implementation challenging, especially for video frames.

## Installation
1. **Clone the repository:**
   ``` bash
    git clone https://github.com/aashish1008/SegmentationModels-CLIPSeg-YOLOv8-DinoSAM.git
    cd SegmentationModels-CLIPSeg-YOLOv8-DinoSAM/GroundingDINO-with-SAM
2. **Install dependencies:**
   ``` bash
   conda create --name groundingdino_sam python=3.10
   conda activate groundingdino_sam
   pip install -r requirements.txt

3. **Setup Grouding Dino repo and install dependencies:**
   ``` bash
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDino
   pip install -e .
4. **Download SAM checkpoint:**
   ``` bash
   cd ..
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## Usage
1. **Prepare your textual prompts and images or video:**
- Ensure you have your textual descriptions and images or video ready for segmentation.

2. **Run the Image Segmentation script:**
   ``` bash
   python ImageSegmentation.py
3. **Run the Video Segmentation script:**
   ``` bash
   python VideoSegmentation.py

## Results
After running the script, the segmentation results will be saved in the output directory. You can visualize the results to see how well the model segmented the image based on the provided textual prompt.

## References
For more details on the underlying research and model architecture, refer to the respective research papers on Grounding DINO and SAM.
**Grounding Dino:** 
- https://arxiv.org/abs/2303.05499
- https://github.com/IDEA-Research/GroundingDINO
**SAM:**
- https://github.com/facebookresearch/segment-anything?tab=readme-ov-file
