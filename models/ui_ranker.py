import asyncio
from enum import Enum
import os
from typing import Optional
from dotenv import load_dotenv

import numpy as np
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.eye_pattern import EyePatternPredictor
from models.utils import get_cropped_icon, evaluate_cropped_icon, chat_anthropic, find_scanning_pattern_scores, draw_bounding_box

load_dotenv()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class UIRanker:
    def __init__(self):
        self.eye_pattern_predictor = EyePatternPredictor()

        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-small')
        self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-small')
        

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


    def calculate_visual_scores(self, contrast_scores, size_scores, tech_savviness):
        """
        Calculate visual scores as weighted average of contrast and size scores.
        Weights vary based on tech savviness:
        - LOW: Size ideality matters more because they need properly-sized, standard UI elements
        - MEDIUM: Balanced consideration of both factors
        - HIGH: Can adapt to varying element sizes, so contrast becomes more important for quick scanning
        """
        if tech_savviness == "LOW":
            size_weight = 0.7     # Less tech-savvy users rely more on standard, properly-sized elements
            contrast_weight = 0.3  # Still need good contrast but size ideality is more crucial
        elif tech_savviness == "MEDIUM":
            size_weight = 0.5     # Equal consideration
            contrast_weight = 0.5
        else:  # HIGH
            size_weight = 0.4     # Can adapt to different sizes as long as they're visible
            contrast_weight = 0.6  # Contrast helps in quick scanning and recognition
        
        return [
            (contrast_weight * c + size_weight * s)
            for c, s in zip(contrast_scores, size_scores)
        ]


    class ScoredElement(BaseModel):
        """Structure to hold both score and reasoning"""
        score: float
        reasoning: str


    def calculate_discoverability_scores(self, screenshot_image, elements, context, last_element):
        age = context["age"]
        tech_savviness = context["tech_savviness"]
        scanning_pattern = self.eye_pattern_predictor.predict(age)
        
        # Calculate base visual scores with reasoning
        contrast_scores = [(score, f"Contrast ratio: {score:.2f}, Age factor applied: {age >= 40}") 
                          for score in self.get_contrast_scores(screenshot_image, elements, age)]
        
        size_scores = [(score, f"Size score: {score:.2f}, Meets accessibility guidelines: {score > 0.6}") 
                       for score in self.get_size_scores(screenshot_image, elements, age)]
        
        # Calculate combined visual scores with reasoning
        visual_scores = []
        for (c_score, c_reason), (s_score, s_reason) in zip(contrast_scores, size_scores):
            score = self.calculate_visual_scores([c_score], [s_score], tech_savviness)[0]
            reasoning = f"Visual prominence: {score:.2f} (Contrast: {c_reason}, Size: {s_reason})"
            visual_scores.append(self.ScoredElement(score=score, reasoning=reasoning))
        
        # Add visual scores to elements for pattern scoring
        for element, visual_score in zip(elements, visual_scores):
            element["visual_score"] = visual_score.score
            element["visual_reasoning"] = visual_score.reasoning
        
        # Get pattern scores with reasoning
        pattern_scores = find_scanning_pattern_scores(elements, scanning_pattern, last_element)
        
        # Combine scores with reasoning
        final_scores = []
        for i, (element, visual_score) in enumerate(zip(elements, visual_scores)):
            pattern_score = pattern_scores[i]
            final_score = 0.4 * pattern_score + 0.6 * visual_score.score
            
            reasoning = f"""
            Final Discoverability: {final_score:.2f}
            - {visual_score.reasoning}
            - Pattern Score: {pattern_score:.2f} ({scanning_pattern} pattern)
            - Tech Savviness: {tech_savviness} influenced visual weight
            """
            
            final_scores.append(self.ScoredElement(score=final_score, reasoning=reasoning.strip()))
        
        return final_scores


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

        class ElementType(str, Enum):
            GENERIC = "generic"
            DOMAIN_SPECIFIC = "domain_specific"

        class UnderstandingLevel(str, Enum):
            LOW = "low"      # Beginner-friendly elements
            MEDIUM = "medium"  # Intermediate-level elements
            HIGH = "high"    # Expert-level elements
    
        # First determine if element is generic or domain-specific
        prompt = """You are an intelligent UI designer. Your task is to determine if the highlighted UI element is:
        
        GENERIC: Common elements found across all UIs (like home buttons, search bars, basic navigation)
        DOMAIN_SPECIFIC: Elements specific to a field/industry (like fiscal quarter input, medical diagnosis codes)
        
        Look at the highlighted element and classify it."""
        
        class GenericOrDomain(BaseModel):
            element_type: ElementType = Field(
                description="GENERIC if element is common across all UIs (home, search, menu). DOMAIN_SPECIFIC if element requires field knowledge (fiscal quarter, diagnosis code)."
            )

        response = await chat_anthropic(prompt, highlighted_image, output_schema=GenericOrDomain)
        
        if response.element_type == ElementType.GENERIC:
            return self.ScoredElement(
                score=1.0,
                reasoning=f"Generic UI element: {element['type']} - Highly understandable across all user levels"
            )
        
        # For domain-specific elements
        prompt = f"""You are an expert in {context['domain_knowledge']} domain. 
        Determine how complex the domain concept behind this UI element is.
        
        Examples:
        LOW: Basic industry terms, common metrics
        MEDIUM: Standard industry calculations, specialized inputs
        HIGH: Complex domain concepts, expert-level metrics"""
        
        class ExpertiseLevel(BaseModel):
            required_level: UnderstandingLevel = Field(
                description=f"""
                LOW: Basic domain concepts (simple industry terms)
                MEDIUM: Standard domain operations (common calculations)
                HIGH: Complex domain concepts (expert-level metrics)
                """
            )
        
        expertise_response = await chat_anthropic(prompt, highlighted_image, output_schema=ExpertiseLevel)
        required_level = expertise_response.required_level
        user_level = context["domain_knowledge_level"]
        
        # Same scoring matrix as before
        scoring_matrix = {
            "high": {
                "high": 1.0,    # Expert with complex concept
                "medium": 1.0,  # Expert with standard concept
                "low": 1.0      # Expert with basic concept
            },
            "medium": {
                "high": 0.7,    # Intermediate with complex concept
                "medium": 1.0,  # Intermediate with standard concept
                "low": 1.0      # Intermediate with basic concept
            },
            "low": {
                "high": 0.2,    # Beginner with complex concept
                "medium": 0.4,  # Beginner with standard concept
                "low": 0.7      # Beginner with basic concept
            }
        }
        
        score = scoring_matrix[user_level][required_level]
        reasoning = f"""
        Domain-specific element: {element['type']}
        - Required expertise: {required_level}
        - User expertise: {user_level}
        - Score: {score:.2f} based on expertise gap
        - Domain: {context['domain_knowledge']}
        """
        
        return self.ScoredElement(score=score, reasoning=reasoning.strip())


    def calculate_semantic_relevance_scores(self, image, elements, task):
        scored_elements = []
        for element in elements:
            cropped_icon = get_cropped_icon(image, element)
            score = evaluate_cropped_icon(cropped_icon, task, self.model, self.tokenizer)
            
            reasoning = f"""
            Semantic Relevance: {score:.2f}
            - Element: {element['type']} ({element.get('text', 'no text')})
            - Task: {task}
            - Relevance: {'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'}
            """
            
            scored_elements.append(self.ScoredElement(score=score, reasoning=reasoning.strip()))
        
        return scored_elements


    def rank_elements(self, screenshot_image, elements, task, context, last_element):
        elements = [
            {
                "element_id": f"{item['type']}_{item['id']}",
                "type": item["type"],
                "text": item["content"],
                "bounds": {
                    "x1": min(item["bbox"][0], item["bbox"][2]),
                    "x2": max(item["bbox"][0], item["bbox"][2]),
                    "y1": min(item["bbox"][1], item["bbox"][3]),
                    "y2": max(item["bbox"][1], item["bbox"][3])
                },
                "position": ((item["bbox"][0] + item["bbox"][2]) / 2, (item["bbox"][1] + item["bbox"][3]) / 2)
            }
            for item in elements
        ]
        
        # elements = self.get_elements_from_image(screenshot_image)
        
        # Discoverability
        discoverability_scores = self.calculate_discoverability_scores(screenshot_image, elements, context, last_element)
        
        # Understandability
        understandability_scores = asyncio.run(asyncio.gather(*[
            self.calculate_understandability_score(element, draw_bounding_box(screenshot_image, element), context) 
            for element in elements
        ]))
        
        # Semantic relevance
        semantic_scores = self.calculate_semantic_relevance_scores(screenshot_image, elements, task)
        
        # Weighted average of the scores
        final_scores = []
        for i, element in enumerate(elements):
            d_score = discoverability_scores[i]
            u_score = understandability_scores[i]
            s_score = semantic_scores[i]
            
            final_score = (
                0.3 * d_score.score +
                0.3 * u_score.score +
                0.4 * s_score.score
            )
            
            reasoning = f"""
            Final Score: {final_score:.2f} for {element['type']}
            
            1. Discoverability ({d_score.score:.2f}):
            {d_score.reasoning}
            
            2. Understandability ({u_score.score:.2f}):
            {u_score.reasoning}
            
            3. Semantic Relevance ({s_score.score:.2f}):
            {s_score.reasoning}
            """
            
            final_scores.append((final_score, reasoning.strip()))
        
        return final_scores


def get_last_interaction(scratchpad):
    """Extract the last element interacted with from the scratchpad"""
    if not scratchpad:
        return None
        
    # Split scratchpad into individual actions
    actions = [action.strip() for action in scratchpad.split('\n') if action.strip()]
    if not actions:
        return None
        
    # Get the last action and extract element info
    last_action = actions[-1]
    return {
        "action": last_action,
        "element_id": None  # You might want to parse this from the action text
    }

def get_next_element(screenshot_image, task, context, scratchpad):
    last_interaction = get_last_interaction(scratchpad)
    if scratchpad:
        prompt = f"""You are a {context["age"]} years old {context["role"]} with {context["domain_knowledge_level"]} {context["domain_knowledge"]} domain expertise and {context["tech_savviness"]} tech savviness, navigating a UI to complete a specific task. 

TASK OBJECTIVE:
```
{task}
```

INTERACTION HISTORY:
You have performed these actions in sequence:
```
{scratchpad}
```
Last interaction: {last_interaction["action"] if last_interaction else "Starting fresh"}

CURRENT SITUATION:
Looking at the current screenshot and considering the interaction history:
1. Does the current page still lead toward the task objective?
2. Have we moved away from the logical path to complete the task?
3. Would a typical {context["role"]} with {context["domain_knowledge_level"]} {context["domain_knowledge"]} domain expertise and {context["tech_savviness"]} tech savviness realize they need to backtrack?

DECISION NEEDED:
Should we continue from this point or backtrack to a previous state?"""

        class BacktrackDecision(BaseModel):
            '''Evaluation of whether to continue or backtrack based on task progress'''
            backtrack: bool = Field(
                description="True if user should return to a previous state (wrong path/dead end), False if current path still leads to objective"
            )
            confidence: float = Field(
                description="Confidence in this decision (0.0-1.0). Consider user's expertise and clarity of the situation",
                ge=0.0,
                le=1.0
            )
            reasoning: str = Field(
                description="Brief explanation of why this decision makes sense for this user profile"
            )

        response = asyncio.run(chat_anthropic(prompt, screenshot_image, output_schema=BacktrackDecision))
        
        if response.backtrack and response.confidence > 0.5:
            return "backtrack", response.reasoning
    
    ui_ranker = UIRanker()    
    return ui_ranker.rank_elements(screenshot_image, task, context, last_interaction)[-1]