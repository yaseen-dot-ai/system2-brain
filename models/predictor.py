import torch
from PIL import Image
import numpy as np
from models.ui_attention_predictor import Platform, UIAttentionPredictor
from models.eye_pattern import EyePatternPredictor
from models.utils import evaluate_cropped_icon, draw_attention, find_next_element_scan, display_image
from models.op_utils.omniparser import Omniparser

from sentence_transformers import SentenceTransformer
import json
import traceback
import matplotlib.pyplot as plt

from dataclasses import dataclass
import os
from dotenv import load_dotenv
import asyncio
import time
import base64
import io
from models.ui_attention_predictor import Platform

load_dotenv()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))


class UIPredictor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eye_pattern_predictor = EyePatternPredictor()
        self.ui_attention_predictor = UIAttentionPredictor()

        op_config = {
            'som_model_path': os.path.join(BASE_PATH, os.getenv('SOM_MODEL_PATH')),
            'caption_model_name': os.getenv('CAPTION_MODEL_NAME', 'florence2'),
            'caption_model_path': os.path.join(BASE_PATH, os.getenv('CAPTION_MODEL_PATH')),
            'BOX_TRESHOLD': float(os.getenv('BOX_THRESHOLD', 0.05)),
        }
        self.omniparser = Omniparser(op_config)

        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.23))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.4))

        # Load the model
        self.embed_model = SentenceTransformer(
            os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2'),
            device=self.device
        )
        
        
    def predict(self, image: Image.Image, age: int, platform: Platform, task: str, tech_saviness: int, debug: bool = False):
        """
        Main prediction function that streams timesteps and handles failures
        """
        try:
            eye_pattern = self.eye_pattern_predictor.predict(age)
            
            # Initial parsing
            _, parsed_content_list = self.omniparser.parse(image)
            
            elements_data = [
                {
                    "type": item["type"],
                    "text": item["content"],
                    "bounds": {
                        "x1": min(item["bbox"][0], item["bbox"][2]),
                        "x2": max(item["bbox"][0], item["bbox"][2]),
                        "y1": min(item["bbox"][1], item["bbox"][3]),
                        "y2": max(item["bbox"][1], item["bbox"][3])
                    }
                }
                for item in parsed_content_list
            ]

            # Get prediction
            result = self.ui_attention_predictor.predict_attention(
                platform=platform,
                tech_savv=tech_saviness,
                ui_image=image,
                task=task,
                elements_data=elements_data
            )
            
            if debug:
                print(json.dumps(result, indent=4))
                attention_image = self.ui_attention_predictor.visualize_attention(result, image, alpha=0.9)
                display_image(attention_image, jup=False)
            
            # Stream each timestep individually
            for timestep in self._generate_timesteps(
                platform,
                image,
                task,
                eye_pattern,
                result,
                debug
            ):
                yield timestep
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            import traceback
            error_trace = traceback.format_exc()
            print(f"Full traceback: {error_trace}")
            yield {
                "status": "error", 
                "message": str(e),
                "traceback": error_trace
            }
            return
    
    
    def _generate_timesteps(self, platform, image, task, eye_pattern, result, debug):
        """
        Internal method to generate timesteps
        This is where your main notebook logic will go
        """
        def image_to_base64(img):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        print(f"attention points:")
        print(json.dumps(result["attention_distribution"], indent=4))

        elements_ref = result["attention_distribution"].copy()
        assert all(elements_ref[i]["element_id"] != elements_ref[i+1]["element_id"] for i in range(len(elements_ref)-1))
        len_elements_ref = len(elements_ref)
        elements = elements_ref.copy()
        last_element = None
        
        scan = False
        scan_pattern = "confidence"
        last_image = image
        
        timestep_count = 0  # Add a counter to track timesteps
        visual_elements = []
        while elements:
            print("#########################")
            curr_window = elements[:5]
            
            scores = np.array([point["score"] for point in curr_window])
            exp_scores = np.exp(scores - np.max(scores))  # Softmax and subtract max for numerical stability
            confidences = exp_scores / exp_scores.sum()    
            
            print(f"confidences: {[f'{conf:.2f}' for conf in confidences]}")
            if scan:
                print(f"scanning {eye_pattern}..")
                element = find_next_element_scan(elements_ref, eye_pattern, last_element, debug)
            else:
                if confidences[0] < self.confidence_threshold:
                    # means the user is not confident about the icon, use scanning eye_pattern to find the icon
                    print(f"not confident, switching to scanning eye_pattern: {eye_pattern}")
                    elements_ref = [point for point in elements_ref if point["candidate_type"] != "platform_hotspot"]
                    print("elements_ref: ", len(elements_ref))
                    elements = elements_ref.copy()
                    scan = True
                    scan_pattern = eye_pattern.lower()
                    last_element = None
                    continue  # Skip the rest of the loop and start scanning
                
                print("guessing next element..")
                if last_element:
                    elements.remove(last_element)
                element = elements[0]
            
            if not element:
                print("no element found..")
                break
            
            # Only visualize and append element if we're confident or in scanning mode
            highlighted_image, color = draw_attention(element, last_image, timestep_count, len_elements_ref)
            highlighted_element = {**element, "color": color, "scan_pattern": scan_pattern}
            print(f"\nhighlighted element: {json.dumps(highlighted_element, indent=4)}")
            visual_elements.append(highlighted_element)
            
            timestep_count += 1  # Increment counter
            
            image_width, image_height = image.size
            x, y = element["position"]
            
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(x * image_width)
            pixel_y = int(y * image_height)
            
            # Calculate bounding box size (15% of image width)
            ratio = {
                Platform.ANDROID: 0.15,
                Platform.IOS: 0.15,
                Platform.DESKTOP: 0.04
            }
            
            box_size = int(ratio[platform] * image_width)
            
            # Calculate initial crop coordinates
            x1 = pixel_x - box_size // 2
            y1 = pixel_y - box_size // 2
            
            # Adjust coordinates if they go beyond image bounds while maintaining square shape
            if x1 < 0:
                x1 = 0
                x2 = box_size
            elif x1 + box_size > image_width:
                x2 = image_width
                x1 = image_width - box_size
            else:
                x2 = x1 + box_size
                
            if y1 < 0:
                y1 = 0
                y2 = box_size
            elif y1 + box_size > image_height:
                y2 = image_height
                y1 = image_height - box_size
            else:
                y2 = y1 + box_size
            
            # Convert PIL Image to numpy array, crop, and convert back to PIL Image
            image_array = np.array(image)
            cropped_array = image_array[y1:y2, x1:x2, :]  # Include all channels
            cropped_icon = Image.fromarray(cropped_array)
            
            if evaluate_cropped_icon(
                self.omniparser.caption_model_processor["model"], 
                self.omniparser.caption_model_processor["processor"], 
                self.embed_model, 
                cropped_icon, 
                task,
                self.similarity_threshold
            ):
                print("task completed, element found..")
                yield {
                    "status": "success",
                    "elements": visual_elements,
                    "timestep": image_to_base64(highlighted_image)
                }
                break
            
            print("task not completed, element not found..")
            last_element = element
            last_image = highlighted_image
            
            # Convert the image to base64 before yielding
            yield {
                "status": "success",
                "elements": visual_elements,
                "timestep": image_to_base64(highlighted_image)
            }


if __name__ == "__main__":
    predictor = UIPredictor()
    image = Image.open("images/ios_home.png")
    for step in predictor.predict(image, 25, Platform.IOS, "find the time", 9, debug=True):
        if step["status"] == "error":
            print(step["message"])
            print(step["traceback"])
        else:
            display_image(step["timestep"], jup=False)