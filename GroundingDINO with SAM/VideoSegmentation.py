import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch

from huggingface_hub import hf_hub_download

# Grounding DINO imports
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, predict

# Segment Anything Model imports
from segment_anything import build_sam, SamPredictor

# Add the GroundingDINO directory to the system path
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))


class VideoSegmentation:
    def __init__(self, device='cuda', ckpt_repo_id="ShilongLiu/GroundingDINO",
                 ckpt_filename="groundingdino_swinb_cogcoor.pth", ckpt_config_filename="GroundingDINO_SwinB.cfg.py"):
        # Initialize the device (GPU if available, else CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Set the checkpoint repository and filenames for Grounding DINO
        self.ckpt_repo_id = ckpt_repo_id
        self.ckpt_filename = ckpt_filename
        self.ckpt_config_filename = ckpt_config_filename

        # Set the SAM checkpoint filename
        self.sam_checkpoint = 'sam_vit_h_4b8939.pth'

        # Initialize the models (SAM and Grounding DINO)
        self.predictor, self.grounding_model = self.initialize_models()

    def load_dino_model_hf(self, repo_id, filename, ckpt_config_filename):
        # Download and load Grounding DINO model configuration
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        args.device = self.device

        # Build and load Grounding DINO model with checkpoint
        model = build_model(args)
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=self.device)
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        model.eval()
        return model

    def initialize_models(self):
        # Initialize SAM model and predictor
        sam = build_sam(checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)

        # Load Grounding DINO model
        model = self.load_dino_model_hf(self.ckpt_repo_id, self.ckpt_filename, self.ckpt_config_filename)

        return predictor, model

    def preprocess_image(self, image_bgr):
        # Preprocess the input image
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    def get_grounding_detect(self, image, caption, box_threshold, text_threshold):
        # Get grounding detection from the model
        processed_image = self.preprocess_image(image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
        return annotated_frame[..., ::-1], boxes  # Convert from BGR to RGB

    def segment(self, image, boxes):
        # Segment the detected boxes in the image
        self.predictor.set_image(image)
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks.cpu()

    def draw_mask(self, mask, image, random_color=False):
        # Draw masks on the image
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0) if random_color else np.array(
            [30 / 255, 144 / 255, 255 / 255, 0.6])
        mask_image = mask.reshape(*mask.shape[-2:], 1) * color.reshape(1, 1, -1)
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image.numpy() * 255).astype(np.uint8)).convert("RGBA")
        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

    def grouding_sam_output(self, frame, caption, box_threshold, text_threshold):
        # Generate grounding SAM output
        annotated_frame, detected_boxes = self.get_grounding_detect(frame, caption, box_threshold, text_threshold)

        masks = self.segment(frame, detected_boxes)
        detection_with_mask = Image.fromarray(annotated_frame).convert("RGBA")

        for i in range(masks.shape[0]):  # Iterate over the number of masks
            for j in range(masks.shape[1]):  # Iterate over the channels of each mask
                mask = masks[i][j]
                detection_with_mask = self.draw_mask(mask, np.array(detection_with_mask))
        return detection_with_mask

    def run(self, video_path, caption, box_threshold, text_threshold):
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Create video writer
        out = cv2.VideoWriter("sheep_segmentation.mp4", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame for segmentation
            detection_with_mask = self.grouding_sam_output(frame, caption, box_threshold, text_threshold)
            out.write(cv2.cvtColor(detection_with_mask, cv2.COLOR_BGR2RGB))
            cv2.imshow('frame', detection_with_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def main():
        # Main function to run video segmentation
        video_path = "video_samples/sheep.mp4"
        caption = "sheep"
        box_threshold = 0.35
        text_threshold = 0.25
        video_segmentation = VideoSegmentation()
        video_segmentation.run(video_path, caption, box_threshold, text_threshold)


# Entry point
if __name__ == "__main__":
    VideoSegmentation.main()
