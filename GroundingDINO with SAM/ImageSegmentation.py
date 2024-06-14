import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import torch

from huggingface_hub import hf_hub_download

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict, Model

import supervision as sv

# Segment Anything
from segment_anything import build_sam, SamPredictor


sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))


def load_dino_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def grounding_sam(device):
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    predictor = sam_predictor(sam_checkpoint, device)

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    grounding = load_dino_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)

    return predictor, grounding


def get_grounding_detect(image_path, text_prompt, model, box_threshold, text_threshold):
    image_source, image_tensor = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    return annotated_frame, boxes


def sam_predictor(sam_checkpoint, device):
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return predictor


def segment(image, sam_model, boxes, device):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.cpu()


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def image_segmentation(image_path, prompt, box_threshold, text_threshold, predictor, model, device):
    image_source, image_tensor = load_image(image_path)

    annotate_image, detected_boxes = get_grounding_detect(image_path, prompt, model, box_threshold, text_threshold)

    masks = segment(image_source, predictor, boxes=detected_boxes, device=device)

    detection_with_mask = Image.fromarray(annotate_image).convert("RGBA")

    for i in range(masks.shape[0]):  # Iterate over the number of masks
        for j in range(masks.shape[1]):  # Iterate over the channels of each mask
            mask = masks[i][j]
            detection_with_mask = draw_mask(mask, np.array(detection_with_mask), random_color=True)

    out_frame = cv2.cvtColor(detection_with_mask, cv2.COLOR_BGR2RGB)

    cv2.imshow("Image", out_frame)
    cv2.imwrite("ImageSeg_Outcome/segmented_image.jpeg", out_frame)

    cv2.waitKey(0)


def main():
    image_path = "image_samples/sheep1_0.jpg"
    prompt = "sheep"
    box_threshold = 0.3
    text_threshold = 0.25
    # Verification of system usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predictor, grounding_model = grounding_sam(device)
    image_segmentation(image_path, prompt, box_threshold, text_threshold, predictor, grounding_model, device)


if __name__ == "__main__":
    main()
