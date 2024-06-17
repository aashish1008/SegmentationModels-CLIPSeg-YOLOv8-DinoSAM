# CLIPSeg: Zero-Shot Image Segmentation

**Features**
-** Text-Guided Segmentation:** Generates segmentation masks based on textual prompts.
- **Zero-Shot Learning:** Performs segmentation without needing extensive annotated training data.
-** Seamless Integration:** Bridges the gap between natural language processing and computer vision.
**Limitations**
- Optimized for static image segmentation and less effective for dynamic video frames.

## Installation
To set up CLIPSeg for Image Segmentation, follow these steps:

1. Clone the repository:
   ``` bash
   git clone https://github.com/aashish1008/SegmentationModels-CLIPSeg-YOLOv8-DinoSAM.git
   cd SegmentationModels-CLIPSeg-YOLOv8-DinoSAM/CLIPSeg
2. Install dependencies:
   It's recommended to use a virtual environment. You can set up a virtual environment using venv or conda.
  For `venv`:
     ``` bash
     python3 -m venv env
     source env/bin/activate  # On Windows, use `env\Scripts\activate`
     pip install -r requirements.txt

  For `conda`:
    ``` bash
    conda create --name clipseg python=3.10
    conda activate clipseg
    pip install -r requirements.txt
     

      

## Uasge

**1. Prepare your textual prompts and images:**
  Ensure you have your textual descriptions and images ready for segmentation.
**2. Run the segmentation script:**
    ``` bash
    python zero_shot_image_segementation.py

## Results
You can visualize the results to see how well the model segmented the image based on the provided textual prompt.
