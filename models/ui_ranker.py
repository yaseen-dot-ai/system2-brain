import asyncio
from enum import Enum
import os
from typing import Optional
from dotenv import load_dotenv

import numpy as np
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.eye_pattern import EyePatternPredictor
from models.op_utils.omniparser import Omniparser
from models.utils import get_cropped_icon, evaluate_cropped_icon, chat_anthropic, find_scanning_pattern_scores, draw_bounding_box

load_dotenv()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class UIRanker:
    def __init__(self):
        self.eye_pattern_predictor = EyePatternPredictor()

        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
        self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
        
        op_config = {
            'som_model_path': os.path.join(BASE_PATH, os.getenv('SOM_MODEL_PATH')),
            'caption_model_name': os.getenv('CAPTION_MODEL_NAME', 'florence2'),
            'caption_model_path': os.path.join(BASE_PATH, os.getenv('CAPTION_MODEL_PATH')),
            'BOX_TRESHOLD': float(os.getenv('BOX_THRESHOLD', 0.05)),
        }
        
        self.omniparser = Omniparser(op_config)
        

    # def get_elements_from_image(self, screenshot_image):
    #     op_config = {
    #         'som_model_path': os.path.join(BASE_PATH, os.getenv('SOM_MODEL_PATH')),
    #         'caption_model_name': os.getenv('CAPTION_MODEL_NAME', 'florence2'),
    #         'caption_model_path': os.path.join(BASE_PATH, os.getenv('CAPTION_MODEL_PATH')),
    #         'BOX_TRESHOLD': float(os.getenv('BOX_THRESHOLD', 0.05)),
    #     }
    #     omniparser = Omniparser(op_config)
    #     _, parsed_content_list = omniparser.parse(screenshot_image)
    #     elements_data = [
    #         {
    #             "type": item["type"],
    #             "text": item["content"],
    #             "bounds": {
    #                 "x1": min(item["bbox"][0], item["bbox"][2]),
    #                 "x2": max(item["bbox"][0], item["bbox"][2]),
    #                 "y1": min(item["bbox"][1], item["bbox"][3]),
    #                 "y2": max(item["bbox"][1], item["bbox"][3])
    #             }
    #         }
    #         for item in parsed_content_list
    #     ]
    #     return elements_data
    

    def get_contrast_scores(self, image, elements, age):
        """Calculate contrast scores for UI elements based on their brightness relative to surroundings.
        Higher scores indicate better contrast which is especially important for older users.
        The contrast importance increases gradually with age starting from 40 years old.
        For older users, low contrast elements are penalized with a practical scaling factor.
        """
        scores = []
        # Convert screenshot to grayscale for luminance calculation
        screenshot_gray = image.convert('L')
        screenshot_array = np.array(screenshot_gray)
        width, height = image.size
        
        # Define age-based contrast importance parameters
        MIN_AGE_EFFECT = 40  # Age at which contrast starts becoming more important
        MAX_AGE_EFFECT = 80  # Age at which contrast importance peaks
        MAX_IMPORTANCE_FACTOR = 0.6  # Maximum power factor increase for contrast importance
        
        for element in elements:
            bounds = element["bounds"]
            # Convert relative coordinates to absolute pixels
            x1 = int(bounds["x1"] * width)
            x2 = int(bounds["x2"] * width)
            y1 = int(bounds["y1"] * height)
            y2 = int(bounds["y2"] * height)
            
            # Get element region
            element_region = screenshot_array[y1:y2, x1:x2]
            element_brightness = np.mean(element_region)
            
            # Get surrounding region (padding of 10 pixels)
            pad = 10
            surr_y1 = max(0, y1 - pad)
            surr_y2 = min(screenshot_array.shape[0], y2 + pad)
            surr_x1 = max(0, x1 - pad)
            surr_x2 = min(screenshot_array.shape[1], x2 + pad)
            
            # Create a mask to exclude the element region
            surr_region = screenshot_array[surr_y1:surr_y2, surr_x1:surr_x2]
            mask = np.ones_like(surr_region, dtype=bool)
            mask[y1-surr_y1:y2-surr_y1, x1-surr_x1:x2-surr_x1] = False
            surrounding_brightness = np.mean(surr_region[mask])
            
            # Calculate contrast ratio
            brighter = max(element_brightness, surrounding_brightness)
            darker = min(element_brightness, surrounding_brightness)
            contrast_ratio = (brighter + 0.05) / (darker + 0.05)
            
            # Normalize to 0-1 range (typical contrast ratios are between 1 and 21)
            score = min(contrast_ratio / 21.0, 1.0)
            
            # Calculate age-based importance factor using a smooth curve
            if age >= MIN_AGE_EFFECT:
                age_factor = min(1.0, (age - MIN_AGE_EFFECT) / (MAX_AGE_EFFECT - MIN_AGE_EFFECT))
                importance_factor = 1.0 + (MAX_IMPORTANCE_FACTOR * age_factor)
                score = score ** importance_factor
            
            scores.append(score)
            
        return scores


    def get_size_scores(self, image, elements, age):
        """Calculate size scores for UI elements based on their dimensions.
        Higher scores are given to elements that are reasonably sized and clearly visible.
        For older users, smaller elements are penalized more to favor larger, easier-to-see elements.
        Considers both relative screen size and absolute minimum sizes for accessibility.
        """
        scores = []
        width, height = image.size
        screen_area = width * height
        
        # Define age-based size importance parameters
        MIN_AGE_EFFECT = 45  # Age at which size starts becoming more important
        MAX_AGE_EFFECT = 75  # Age at which size importance peaks
        MAX_IMPORTANCE_FACTOR = 1.2  # Maximum power factor increase for size importance
        
        # Define reasonable size bounds (as percentage of screen area)
        MIN_IDEAL_SIZE_RATIO = 0.008  # Minimum ideal size (0.8% of screen)
        MAX_IDEAL_SIZE_RATIO = 0.12   # Maximum ideal size (12% of screen)
        
        # Define absolute minimum size in pixels (useful for very high-res screens)
        MIN_PIXELS = 48 * 48  # Minimum touchable area as per accessibility guidelines
        
        for element in elements:
            bounds = element["bounds"]
            # Convert relative coordinates to absolute pixels
            element_width = int((bounds["x2"] - bounds["x1"]) * width)
            element_height = int((bounds["y2"] - bounds["y1"]) * height)
            element_area = element_width * element_height
            
            # Calculate size ratio relative to screen
            size_ratio = element_area / screen_area
            
            # Calculate base score considering both relative and absolute size
            if element_area < MIN_PIXELS:
                score = 0.4 + 0.2 * (element_area / MIN_PIXELS) ** 0.5
            elif size_ratio < MIN_IDEAL_SIZE_RATIO:
                ratio_to_min = size_ratio / MIN_IDEAL_SIZE_RATIO
                score = 0.6 + 0.3 * ratio_to_min
            elif size_ratio > MAX_IDEAL_SIZE_RATIO:
                score = 0.9
            else:
                score = 1.0
            
            # Apply age-based adjustments using importance factor
            if age >= MIN_AGE_EFFECT and (element_area < MIN_PIXELS or size_ratio < MIN_IDEAL_SIZE_RATIO):
                age_factor = min(1.0, (age - MIN_AGE_EFFECT) / (MAX_AGE_EFFECT - MIN_AGE_EFFECT))
                importance_factor = 1.0 + (MAX_IMPORTANCE_FACTOR * age_factor)
                
                if element_area < MIN_PIXELS:
                    importance_factor *= 1.2
                
                score = score ** importance_factor
            
            scores.append(score)
        
        return scores


    def get_interactivity_scores(self, elements):
        """
        Calculate interactivity scores for UI elements based on their type and appearance.
        Higher scores for elements that appear more interactive/clickable.
        """
        interactivity_weights = {
            # High interactivity (1.0)
            "button": 1.0,
            "link": 1.0,
            "input": 1.0,
            "select": 1.0,
            "checkbox": 1.0,
            "radio": 1.0,
            
            # Medium interactivity (0.7)
            "dropdown": 0.7,
            "menu": 0.7,
            "tab": 0.7,
            "icon": 0.7,
            
            # Low interactivity (0.4)
            "text": 0.4,
            "label": 0.4,
            "image": 0.4,
            
            # Default for unknown types
            "default": 0.4
        }
        
        scores = []
        for element in elements:
            element_type = element["type"].lower()
            
            # Check for interactive keywords in text
            has_interactive_text = False
            if "text" in element:
                text = element["text"].lower()
                interactive_keywords = ["click", "tap", "select", "choose", "submit", "login", "sign in", "register"]
                has_interactive_text = any(keyword in text for keyword in interactive_keywords)
            
            # Get base score from type
            base_score = interactivity_weights.get(element_type, interactivity_weights["default"])
            
            # Boost score if text suggests interactivity
            if has_interactive_text:
                base_score = min(1.0, base_score + 0.2)
            
            scores.append(base_score)
        
        return scores


    def calculate_visual_scores(self, contrast_scores, size_scores, interactivity_scores, tech_savviness):
        """
        Calculate visual scores as weighted average of contrast, size, and interactivity scores.
        Weights vary based on tech savviness:
        - LOW: Size ideality and interactivity matter more (need obvious interactive elements)
        - MEDIUM: Balanced consideration
        - HIGH: Can recognize subtle interactive elements, so contrast becomes more important
        """
        if tech_savviness == "LOW":
            size_weight = 0.4      # Need properly-sized elements
            contrast_weight = 0.2   # Basic visibility
            interact_weight = 0.4   # Need obvious interactive elements
        elif tech_savviness == "MEDIUM":
            size_weight = 0.3      # Balanced
            contrast_weight = 0.4
            interact_weight = 0.3
        else:  # HIGH
            size_weight = 0.2      # Can adapt to different sizes
            contrast_weight = 0.5   # Quick scanning
            interact_weight = 0.3   # Can recognize subtle interactive elements
        
        return [
            (contrast_weight * c + size_weight * s + interact_weight * i)
            for c, s, i in zip(contrast_scores, size_scores, interactivity_scores)
        ]


    def calculate_delta_scores(self, current_image, previous_image, elements):
        """
        Hybrid approach combining direct pixel differences with structural analysis
        for better change detection.
        """
        import cv2
        import numpy as np
        
        curr_array = np.array(current_image.convert('RGB'))
        prev_array = np.array(previous_image.convert('RGB'))
        
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_array, cv2.COLOR_RGB2GRAY)
        prev_gray = cv2.cvtColor(prev_array, cv2.COLOR_RGB2GRAY)
        
        height, width = curr_gray.shape
        delta_scores = []
        
        for element in elements:
            # Get element bounds
            x1 = int(element["bounds"]["x1"] * width)
            x2 = int(element["bounds"]["x2"] * width)
            y1 = int(element["bounds"]["y1"] * height)
            y2 = int(element["bounds"]["y2"] * height)
            
            curr_region = curr_gray[y1:y2, x1:x2]
            prev_region = prev_gray[y1:y2, x1:x2]
            
            try:
                # 1. Direct pixel differences for large changes
                diff = cv2.absdiff(curr_region, prev_region)
                mse = np.mean(diff ** 2)
                pixel_change = min(1.0, mse / 10000)
                
                # 2. Structural changes for subtle differences
                # Apply Gaussian blur with smaller kernel for finer detail
                blurred_curr = cv2.GaussianBlur(curr_region, (3, 3), 0)
                blurred_prev = cv2.GaussianBlur(prev_region, (3, 3), 0)
                structural_diff = cv2.absdiff(blurred_curr, blurred_prev)
                structural_change = np.mean(structural_diff > 20) # Lower threshold for subtle changes
                
                # Combine scores with emphasis on larger changes
                change_score = max(
                    pixel_change,  # Catches obvious changes
                    structural_change * 0.8  # Catches subtle changes but weighted less
                )
                
            except Exception:
                change_score = 1.0
            
            # Categorize with more granular levels
            if change_score > 0.8:
                reason = "Major change - new or significantly modified element"
            elif change_score > 0.5:
                reason = "Moderate change - content or style updated"
            elif change_score > 0.2:
                reason = "Subtle change - minor updates or style changes"
            else:
                reason = "Minimal or no change"
            
            # Center-screen boost
            center_y = (y1 + y2) / (2 * height)
            center_x = (x1 + x2) / (2 * width)
            center_distance = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            center_boost = max(0, 1 - center_distance)
            
            final_score = min(1.0, change_score * (1 + 0.3 * center_boost))
            delta_scores.append((final_score, reason))
        
        return delta_scores


    def calculate_discoverability_scores(self, screenshot_image, previous_image, elements, context, last_element):
        age = int(context["age"])
        tech_savviness = context["tech_savviness"]
        scanning_pattern = self.eye_pattern_predictor.predict(age)
        
        # Calculate base visual scores
        contrast_scores = self.get_contrast_scores(screenshot_image, elements, age)
        size_scores = self.get_size_scores(screenshot_image, elements, age)
        interactivity_scores = self.get_interactivity_scores(elements)
        visual_scores = self.calculate_visual_scores(
            contrast_scores, 
            size_scores,
            interactivity_scores,
            tech_savviness
        )
        
        # Calculate delta scores if we have previous screenshot
        if previous_image is not None:
            delta_scores = self.calculate_delta_scores(screenshot_image, previous_image, elements)
        else:
            # First view - all elements are "new"
            delta_scores = [(1.0, "Initial view")] * len(elements)
        
        # Get pattern scores
        pattern_scores_dict = {k: v for k, v in find_scanning_pattern_scores(elements, scanning_pattern, last_element)}
        pattern_scores = [pattern_scores_dict[element["element_id"]] for element in elements]
        
        # Combine scores with heavy weight on delta
        for i, (element, visual_score) in enumerate(zip(elements, visual_scores)):
            pattern_score = pattern_scores[i]
            delta_score, delta_reason = delta_scores[i]
            
            # Heavy emphasis on changes (0.4), balanced visual (0.3) and pattern (0.3)
            final_score = (
                0.4 * delta_score +    # Changes grab immediate attention
                0.3 * visual_score +   # Visual prominence
                0.3 * pattern_score    # Scanning pattern influence
            )
            
            reasoning = f"""
            Final Discoverability: {final_score:.2f}
            - Change Detection: {delta_score:.2f} ({delta_reason})
            - Visual Score: {visual_score:.2f} (Contrast: {contrast_scores[i]:.2f}, Size: {size_scores[i]:.2f}, Interactivity: {interactivity_scores[i]:.2f})
            - Pattern Score: {pattern_score:.2f} ({scanning_pattern} pattern)
            - Tech Savviness: {tech_savviness} influenced weights
            """
            
            element["discoverability_score"] = final_score
            element["discoverability_reasoning"] = reasoning.strip()
        
        return


    async def calculate_understandability_score(self, element, highlighted_image, context):
        """
        Calculate understandability score based on:
        1. Whether element is generic (common across all UIs) or domain-specific (requires field knowledge)
        2. For domain-specific elements, how complex is the domain concept
        
        Examples:
        Generic elements:
        - Home button, Back button, Search bar, Menu icon, Settings gear
        - Basic form elements (submit, cancel, checkbox)
        - Navigation elements (tabs, links, breadcrumbs)
        
        Domain-specific elements:
        - Financial: Fiscal quarter input, P/E ratio field, EBITDA calculator
        - Medical: ICD-10 code entry, dosage calculator, diagnosis fields
        - Engineering: Tolerance specification, material property inputs
        """
        from enum import Enum
    
        # First determine if element is generic or domain-specific
        prompt = """You are an intelligent UI designer. Your task is to determine if the highlighted UI element is:
        
        GENERIC: Common elements found across all UIs (like home buttons, search bars, basic navigation)
        DOMAIN_SPECIFIC: Elements specific to a field/industry (like fiscal quarter input, medical diagnosis codes)
        
        Look at the highlighted element and classify it."""
        
        class ElementType(str, Enum):
            GENERIC = "generic"
            DOMAIN_SPECIFIC = "domain_specific"
        
        class GenericOrDomain(BaseModel):
            element_type: ElementType = Field(
                description="GENERIC if element is common across all UIs (home, search, menu). DOMAIN_SPECIFIC if element requires field knowledge (fiscal quarter, diagnosis code)."
            )

        response = await chat_anthropic(prompt, highlighted_image, output_schema=GenericOrDomain)
        
        if response.element_type == ElementType.GENERIC:
            element["understandability_score"] = 1.0
            element["understandability_reasoning"] = f"Generic UI element: {element['type']} - Highly understandable across all user levels"
            return
        
        # For domain-specific elements
        prompt = f"""You are an expert in {context['domain']} domain. 
        Determine how complex the domain concept behind this UI element is.
        
        Examples:
        BEGINNER: Basic industry terms, common metrics
        INTERMEDIATE: Standard industry calculations, specialized inputs
        EXPERT: Complex domain concepts, expert-level metrics"""
        
        class UnderstandingLevel(str, Enum):
            BEGINNER = "beginner"      # Beginner-friendly elements
            INTERMEDIATE = "intermediate"  # Intermediate-level elements
            EXPERT = "expert"    # Expert-level elements
        
        class ExpertiseLevel(BaseModel):
            required_level: UnderstandingLevel = Field(
                description=f"""
                BEGINNER: Basic domain concepts (simple industry terms)
                INTERMEDIATE: Standard domain operations (common calculations)
                EXPERT: Complex domain concepts (expert-level metrics)
                """
            )
        
        expertise_response = await chat_anthropic(prompt, highlighted_image, output_schema=ExpertiseLevel)
        required_level = expertise_response.required_level
        user_level = context["domain_familiarity"].lower()
        
        # Same scoring matrix as before
        scoring_matrix = {
            "expert": {
                "expert": 1.0,    # Expert with complex concept
                "intermediate": 1.0,  # Expert with standard concept
                "beginner": 1.0      # Expert with basic concept
            },
            "intermediate": {
                "expert": 0.7,    # Intermediate with complex concept
                "intermediate": 1.0,  # Intermediate with standard concept
                "beginner": 1.0      # Intermediate with basic concept
            },
            "beginner": {
                "expert": 0.2,    # Beginner with complex concept
                "intermediate": 0.4,  # Beginner with standard concept
                "beginner": 0.7      # Beginner with basic concept
            }
        }
        
        score = scoring_matrix[user_level][required_level]
        reasoning = f"""
        Domain-specific element: {element['type']}
        - Required expertise: {required_level}
        - User expertise: {user_level}
        - Score: {score:.2f} based on expertise gap
        - Domain: {context['domain']}
        """
        
        element["understandability_score"] = score
        element["understandability_reasoning"] = reasoning.strip()
        return


    def calculate_semantic_relevance_scores(self, image, elements, task):
        for element in elements:
            cropped_icon = get_cropped_icon(image, element)
            score = evaluate_cropped_icon(cropped_icon, task, self.model, self.tokenizer, self.omniparser.caption_model_processor)
            
            reasoning = f"""
            Semantic Relevance: {score:.2f}
            - Element: {element['type']} ({element.get('text', 'no text')})
            - Task: {task}
            - Relevance: {'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'}
            """
            
            element["semantic_relevance_score"] = score
            element["semantic_relevance_reasoning"] = reasoning.strip()
        
        return


    async def rank_elements(self, screenshot_image, previous_screenshot_image, elements, task, context, last_element):
        # assert (screenshot_image.width, screenshot_image.height) == (1920, 1080), f"Screenshot must be 1920x1080, got {screenshot_image.width}x{screenshot_image.height}"
        
        elements = [
            {
                "element_id": f"{item.type}-{item.text}-{item.x}-{item.y}-{item.width}-{item.height}",
                "type": item.type,
                "text": item.text,
                "bounds": {
                    "x1": item.x / screenshot_image.width,
                    "x2": (item.x + item.width) / screenshot_image.width,
                    "y1": item.y / screenshot_image.height,
                    "y2": (item.y + item.height) / screenshot_image.height
                },
                "position": (
                    # Find center by averaging left and right edges (normalized to 0-1)
                    (item.x + (item.x + item.width)) / (2 * screenshot_image.width),
                    # Find center by averaging top and bottom edges (normalized to 0-1)
                    (item.y + (item.y + item.height)) / (2 * screenshot_image.height)
                )
            }
            for i, item in enumerate(elements)
        ]
        
        if last_element:
            last_element = {
                "element_id": f"{last_element.type}-{last_element.text}-{last_element.x}-{last_element.y}-{last_element.width}-{last_element.height}",
                "type": last_element.type,
                "text": last_element.text,
                "bounds": {
                    "x1": last_element.x / screenshot_image.width,
                    "x2": (last_element.x + last_element.width) / screenshot_image.width,
                    "y1": last_element.y / screenshot_image.height,
                    "y2": (last_element.y + last_element.height) / screenshot_image.height
                },
                "position": (
                    (last_element.x + (last_element.x + last_element.width)) / (2 * screenshot_image.width),
                    (last_element.y + (last_element.y + last_element.height)) / (2 * screenshot_image.height)
                )
            }
        
        print("Parsed elements...")
        
        # Discoverability
        self.calculate_discoverability_scores(screenshot_image, previous_screenshot_image, elements, context, last_element)
        
        print("Calculated discoverability scores...")
        
        # Understandability
        await asyncio.gather(*[
            self.calculate_understandability_score(element, draw_bounding_box(screenshot_image, element), context) 
            for element in elements
        ])
        
        print("Calculated understandability scores...")
        
        # Semantic relevance
        self.calculate_semantic_relevance_scores(screenshot_image, elements, task)
        
        print("Calculated semantic relevance scores...")
        
        # Weighted average of the scores
        for i, element in enumerate(elements):
            d_score = element["discoverability_score"]
            u_score = element["understandability_score"]
            s_score = element["semantic_relevance_score"]
            
            final_score = (
                0.3 * d_score +
                0.3 * u_score +
                0.4 * s_score
            )
            
            reasoning = f"""
            Final Score: {final_score:.2f} for {element['type']}
            
            1. Discoverability ({d_score:.2f}):
            {element["discoverability_reasoning"]}
            
            2. Understandability ({u_score:.2f}):
            {element["understandability_reasoning"]}
            
            3. Semantic Relevance ({s_score:.2f}):
            {element["semantic_relevance_reasoning"]}
            """
            
            element["final_score"] = final_score
            element["final_reasoning"] = reasoning.strip()
        
        return sorted(elements, key=lambda x: x["final_score"], reverse=True)
