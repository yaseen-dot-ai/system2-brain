from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from enum import Enum
from PIL import Image
import time
import cv2  # For color space conversion


class Platform(Enum):
    ANDROID = "android"
    IOS = "ios"
    DESKTOP = "desktop"


@dataclass
class UIElement:
    id: str
    element_type: str
    bounds: Dict[str, float]
    visual_properties: Dict[str, float]
    platform_specific: Dict[str, bool]

    def center_point(self) -> Tuple[float, float]:
        return (
            (self.bounds["x1"] + self.bounds["x2"]) / 2,
            (self.bounds["y1"] + self.bounds["y2"]) / 2
        )

    def size(self) -> float:
        return (self.bounds["x2"] - self.bounds["x1"]) * (self.bounds["y2"] - self.bounds["y1"])


@dataclass
class AttentionCandidate:
    position: Tuple[float, float]  # x, y coordinates (normalized 0-1)
    candidate_type: str  # 'ui_element' or 'platform_hotspot'
    element_id: Optional[str] = None  # Only for UI elements
    element: Optional[UIElement] = None  # Only for UI elements


class UIAttentionPredictor:
    def __init__(self):
        """
        Initialize the predictor with platform and user tech savviness (1-10)
        """
        
        # Define all possible hotspots with their normalized coordinates and weights
        self.hotspot_definitions = {
            Platform.ANDROID: {
                # System UI elements
                "status_bar": {
                    "bounds": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.05},
                    "weight": 0.9
                },
                "time": {
                    "bounds": {"x1": 0.0, "y1": 0.0, "x2": 0.2, "y2": 0.05},
                    "weight": 0.9
                },
                "battery": {
                    "bounds": {"x1": 0.85, "y1": 0.0, "x2": 1.0, "y2": 0.05},
                    "weight": 0.9
                },
                "wifi": {
                    "bounds": {"x1": 0.75, "y1": 0.0, "x2": 0.85, "y2": 0.05},
                    "weight": 0.9
                },
                "notifications": {
                    "bounds": {"x1": 0.0, "y1": 0.0, "x2": 0.3, "y2": 0.05},
                    "weight": 0.9
                },
                
                # Navigation elements
                "bottom_nav": {
                    "bounds": {"x1": 0.3, "y1": 0.95, "x2": 0.7, "y2": 1.0},
                    "weight": 0.8
                },
                "back_button_shortcut": {
                    "bounds": {"x1": 0.25, "y1": 0.95, "x2": 0.35, "y2": 1.0},
                    "weight": 0.9
                },
                "menu_button_shortcut": {
                    "bounds": {"x1": 0.45, "y1": 0.95, "x2": 0.55, "y2": 1.0},
                    "weight": 0.8
                },
                "home_button_shortcut": {
                    "bounds": {"x1": 0.45, "y1": 0.95, "x2": 0.55, "y2": 1.0},
                    "weight": 0.8
                },
                "back_button_ui": {
                    "bounds": {"x1": 0.15, "y1": 0.05, "x2": 0.25, "y2": 0.15},
                    "weight": 0.9
                },
                "menu_button_ui": {
                    "bounds": {"x1": 0.75, "y1": 0.05, "x2": 0.85, "y2": 0.15},
                    "weight": 0.8
                }
            },
            
            Platform.IOS: {
                # System UI elements
                "status_bar": {
                    "bounds": {"x1": 0.0, "y1": 0.95, "x2": 1.0, "y2": 1.0},
                    "weight": 0.9
                },
                "dynamic_island": {
                    "bounds": {"x1": 0.4, "y1": 0.95, "x2": 0.6, "y2": 1.0},
                    "weight": 0.9
                },
                "time": {
                    "bounds": {"x1": 0.45, "y1": 0.95, "x2": 0.55, "y2": 1.0},
                    "weight": 0.9
                },
                "battery": {
                    "bounds": {"x1": 0.85, "y1": 0.95, "x2": 0.95, "y2": 1.0},
                    "weight": 0.9
                },
                "wifi": {
                    "bounds": {"x1": 0.75, "y1": 0.95, "x2": 0.85, "y2": 1.0},
                    "weight": 0.9
                },
                "notifications": {
                    "bounds": {"x1": 0.95, "y1": 0.95, "x2": 1.0, "y2": 1.0},
                    "weight": 0.9
                },
                
                # Navigation elements
                "back_gesture": {
                    "bounds": {"x1": 0.0, "y1": 0.0, "x2": 0.1, "y2": 1.0},
                    "weight": 0.9
                },
                "action_buttons": {
                    "bounds": {"x1": 0.85, "y1": 0.85, "x2": 1.0, "y2": 0.95},
                    "weight": 0.8
                },
                "pull_down": {
                    "bounds": {"x1": 0.0, "y1": 0.9, "x2": 1.0, "y2": 1.0},
                    "weight": 0.8
                },
                "home_indicator": {
                    "bounds": {"x1": 0.4, "y1": 0.0, "x2": 0.6, "y2": 0.05},
                    "weight": 0.8
                },
                "tab_bar": {
                    "bounds": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.1},
                    "weight": 0.8
                }
            },
            
            Platform.DESKTOP: {
                # System UI elements
                "menu_bar": {
                    "bounds": {"x1": 0.0, "y1": 0.95, "x2": 1.0, "y2": 1.0},
                    "weight": 0.9
                },
                "system_tray": {
                    "bounds": {"x1": 0.9, "y1": 0.95, "x2": 1.0, "y2": 1.0},
                    "weight": 0.9
                },
                "time": {
                    "bounds": {"x1": 0.85, "y1": 0.95, "x2": 0.95, "y2": 1.0},
                    "weight": 0.9
                },
                "battery": {
                    "bounds": {"x1": 0.85, "y1": 0.95, "x2": 0.95, "y2": 1.0},
                    "weight": 0.9
                },
                "wifi": {
                    "bounds": {"x1": 0.85, "y1": 0.95, "x2": 0.95, "y2": 1.0},
                    "weight": 0.9
                },
                "notifications": {
                    "bounds": {"x1": 0.8, "y1": 0.7, "x2": 0.9, "y2": 0.8},
                    "weight": 0.9
                },
                
                # Navigation elements
                "start_menu": {
                    "bounds": {"x1": 0.3, "y1": 0.85, "x2": 0.45, "y2": 0.95},
                    "weight": 0.9
                },
                "nav_sidebar": {
                    "bounds": {"x1": 0.9, "y1": 0.15, "x2": 1.0, "y2": 0.85},
                    "weight": 0.7
                },
                "secondary_sidebar": {
                    "bounds": {"x1": 0.8, "y1": 0.15, "x2": 0.9, "y2": 0.85},
                    "weight": 0.6
                }
            }
        }

        self.distance_threshold = {
            Platform.ANDROID: 0.05,
            Platform.IOS: 0.05,
            Platform.DESKTOP: 0.05
        }

        self._task_score_cache = {}  # Simple cache for task scores


    def _calculate_visual_attraction_score(self, candidate: AttentionCandidate) -> float:
        """
        Calculate how visually attractive a point is to the eye.
        For UI elements, considers their visual properties.
        For platform hotspots (no UI element), returns 0.
        """
        if candidate.candidate_type != 'ui_element':
            return 0.0
        
        element = candidate.element
        
        # Visual property weights
        weights = {
            "size": 0.3,        # Larger elements draw more attention
            "contrast": 0.3,    # High contrast elements pop out
            "color": 0.2,      # Bright/saturated colors attract attention
            "motion": 0.1,     # Animated elements draw eye movement
            "isolation": 0.1   # Elements with more whitespace around them stand out
        }
        
        scores = {
            "size": element.visual_properties.get("size", 0.0),
            "contrast": element.visual_properties.get("contrast", 0.0),
            "color": element.visual_properties.get("color_intensity", 0.0),
            "motion": element.visual_properties.get("is_animated", 0.0),
            "isolation": element.visual_properties.get("whitespace", 0.0)
        }
        
        visual_score = sum(weights[k] * scores[k] for k in weights.keys())
        
        # Inverse tech savviness modifier for visual attraction
        # Less tech-savvy users rely more on visual properties
        tech_factor = (11 - self.tech_savv) / 5.0  # 1-10 scale becomes 2.0-0.2
        
        return visual_score * tech_factor
    
    
    def _calculate_base_position_score(self, position: Tuple[float, float]) -> float:
        """
        Calculate base position score for any candidate based on platform conventions
        and center-screen bias using the hotspot definitions.
        """
        x, y = position
        platform_hotspots = self.hotspot_definitions[self.platform]
        max_score = 0.0
        
        # Calculate center-screen bias
        center_x, center_y = 0.5, 0.5  # Screen center in normalized coordinates
        distance_to_center = np.sqrt(
            (x - center_x)**2 + 
            (y - center_y)**2
        )
        # Convert distance to score (1 at center, 0 at corners)
        max_center_distance = np.sqrt(0.5**2 + 0.5**2)  # Distance from center to corner
        center_score = 1 - (distance_to_center / max_center_distance)
        
        # Check each hotspot definition
        for hotspot_name, hotspot_data in platform_hotspots.items():
            bounds = hotspot_data["bounds"]
            weight = hotspot_data["weight"]
            
            # Check if position is within hotspot bounds
            if (bounds["x1"] <= x <= bounds["x2"] and 
                bounds["y1"] <= y <= bounds["y2"]):
                # Calculate distance from center of hotspot
                center_x = (bounds["x1"] + bounds["x2"]) / 2
                center_y = (bounds["y1"] + bounds["y2"]) / 2
                distance = np.sqrt(
                    (x - center_x)**2 + 
                    (y - center_y)**2
                )
                
                # Calculate score based on distance from center
                # Closer to center = higher score
                max_distance = np.sqrt(
                    (bounds["x2"] - bounds["x1"])**2 + 
                    (bounds["y2"] - bounds["y1"])**2
                ) / 2
                distance_score = 1 - (distance / max_distance)
                
                # Combine with hotspot weight
                score = distance_score * weight
                max_score = max(max_score, score)
        
        # Apply tech savviness modifier to hotspot score
        tech_factor = self.tech_savv / 5.0  # 1-10 scale becomes 0.2-2.0
        hotspot_score = max_score * tech_factor
        
        # Combine hotspot score with center bias
        # Center bias weight increases as tech savviness decreases
        center_weight = 0.4 + (0.2 * (11 - self.tech_savv) / 10)  # 0.4-0.6 range
        hotspot_weight = 1 - center_weight
        
        return (hotspot_score * hotspot_weight) + (center_score * center_weight)


    def _calculate_base_task_score(self, position: Tuple[float, float], task: str) -> float:
        """Calculate task-based score using hotspot definitions and caching"""
        cache_key = (position, task.lower())
        if cache_key in self._task_score_cache:
            return self._task_score_cache[cache_key]
        
        x, y = position
        task_lower = task.lower()
        platform_hotspots = self.hotspot_definitions[self.platform]
        max_score = 0.0  # Default score

        # Check each hotspot for task relevance
        for hotspot_name, hotspot_data in platform_hotspots.items():
            bounds = hotspot_data["bounds"]
            weight = hotspot_data["weight"]
            
            # Skip if position is not in this hotspot
            if not (bounds["x1"] <= x <= bounds["x2"] and 
                   bounds["y1"] <= y <= bounds["y2"]):
                continue
            
            # Calculate distance from center
            center_x = (bounds["x1"] + bounds["x2"]) / 2
            center_y = (bounds["y1"] + bounds["y2"]) / 2
            distance = np.sqrt(
                (x - center_x)**2 + 
                (y - center_y)**2
            )
            
            # Calculate base score for this hotspot
            hotspot_score = 0.0
            
            # Check hotspot-specific task keywords
            if hotspot_name == "time" and any(status in task_lower for status in ["time", "clock", "hour"]):
                hotspot_score = 0.9
            elif hotspot_name == "battery" and any(status in task_lower for status in ["battery", "charge", "power"]):
                hotspot_score = 0.9
            elif hotspot_name == "wifi" and any(status in task_lower for status in ["wifi", "network", "signal", "connection"]):
                hotspot_score = 0.9
            elif hotspot_name == "notifications" and any(status in task_lower for status in ["notification", "alert", "message"]):
                hotspot_score = 0.9
            elif hotspot_name == "status_bar" and any(status in task_lower for status in ["status", "system"]):
                hotspot_score = 0.8
            elif hotspot_name == "dynamic_island" and any(status in task_lower for status in ["dynamic", "island", "notch"]):
                hotspot_score = 0.8
            elif hotspot_name == "system_tray" and any(status in task_lower for status in ["tray", "system"]):
                hotspot_score = 0.8
            elif hotspot_name == "menu_bar" and any(status in task_lower for status in ["menu", "file"]):
                hotspot_score = 0.8
            
            # Platform-specific navigation tasks
            if self.platform == Platform.ANDROID:
                if hotspot_name == "back_button" and any(nav in task_lower for nav in ["back", "previous"]):
                    hotspot_score = 0.9
                elif hotspot_name == "menu_button" and any(nav in task_lower for nav in ["menu", "options"]):
                    hotspot_score = 0.8
                elif hotspot_name == "fab" and any(nav in task_lower for nav in ["add", "create"]):
                    hotspot_score = 0.8
                elif hotspot_name.startswith("bottom_nav") and any(nav in task_lower for nav in ["home", "search", "profile"]):
                    hotspot_score = 0.7
            
            elif self.platform == Platform.IOS:
                if hotspot_name == "back_gesture" and any(nav in task_lower for nav in ["back", "previous"]):
                    hotspot_score = 0.9
                elif hotspot_name == "action_buttons" and any(nav in task_lower for nav in ["share", "action"]):
                    hotspot_score = 0.8
                elif hotspot_name == "pull_down" and any(nav in task_lower for nav in ["notification", "search"]):
                    hotspot_score = 0.8
                elif hotspot_name == "home_indicator" and any(nav in task_lower for nav in ["home", "switch"]):
                    hotspot_score = 0.8
                elif hotspot_name == "tab_bar" and any(nav in task_lower for nav in ["tab", "navigate"]):
                    hotspot_score = 0.7
            
            elif self.platform == Platform.DESKTOP:
                if hotspot_name == "main_menu" and any(nav in task_lower for nav in ["menu", "file"]):
                    hotspot_score = 0.9
                elif hotspot_name == "user_menu" and any(nav in task_lower for nav in ["profile", "account", "settings"]):
                    hotspot_score = 0.8
                elif hotspot_name == "nav_sidebar" and any(nav in task_lower for nav in ["navigation", "sidebar"]):
                    hotspot_score = 0.7
                elif hotspot_name == "secondary_sidebar" and any(nav in task_lower for nav in ["details", "properties"]):
                    hotspot_score = 0.6
            
            # Apply distance-based scaling
            max_distance = np.sqrt(
                (bounds["x2"] - bounds["x1"])**2 + 
                (bounds["y2"] - bounds["y1"])**2
            ) / 2
            distance_score = 1 - (distance / max_distance)
            
            # Combine scores
            final_hotspot_score = hotspot_score * distance_score * weight
            max_score = max(max_score, final_hotspot_score)
        
        # Apply tech savviness modifier
        tech_factor = self.tech_savv / 5.0 # 1-10 scale becomes 0.2-2.0
        
        self._task_score_cache[cache_key] = max_score * tech_factor
        return self._task_score_cache[cache_key]


    def _merge_overlapping_candidates(self, candidates: List[AttentionCandidate], proximity_threshold: float = 0.15) -> List[AttentionCandidate]:
        """
        Merge or filter candidates that are too close to each other.
        Prefer UI elements over platform hotspots when there's overlap.
        proximity_threshold: normalized distance (0-1) to consider candidates as overlapping
        """
        merged_candidates = []
        
        # Sort candidates to process UI elements first
        sorted_candidates = sorted(
            candidates,
            key=lambda c: 0 if c.candidate_type == 'ui_element' else 1
        )
        
        for candidate in sorted_candidates:
            # Check if this candidate is too close to any existing merged candidate
            is_redundant = False
            for existing in merged_candidates:
                distance = np.sqrt(
                    (candidate.position[0] - existing.position[0])**2 +
                    (candidate.position[1] - existing.position[1])**2
                )
                
                if distance < proximity_threshold:
                    # If current is UI element and existing is hotspot, replace existing
                    if (candidate.candidate_type == 'ui_element' and 
                        existing.candidate_type == 'platform_hotspot'):
                        merged_candidates.remove(existing)
                        merged_candidates.append(candidate)
                    # If both are UI elements or current is hotspot, skip current
                    is_redundant = True
                    break
            
            if not is_redundant:
                merged_candidates.append(candidate)
        
        return merged_candidates


    def _extract_ui_elements_from_image(self, ui_image: Image, elements_data: List[Dict]) -> List[AttentionCandidate]:
        """
        Convert provided UI elements data into attention candidates with visual properties.
        
        Args:
            ui_image: PIL Image of the UI screenshot
            elements_data: List of dictionaries containing:
                - type: str (button, text, icon, etc)
                - text: str (if any)
                - bounds: Dict with x1,y1,x2,y2 in normalized coordinates
        
        Returns:
            List of AttentionCandidate objects with computed visual properties
        """
        candidates = []
        img_array = np.array(ui_image)
        
        for idx, element in enumerate(elements_data):
            # Convert normalized bounds to pixel coordinates
            x1, y1 = int(element['bounds']['x1'] * ui_image.width), int(element['bounds']['y1'] * ui_image.height)
            x2, y2 = int(element['bounds']['x2'] * ui_image.width), int(element['bounds']['y2'] * ui_image.height)
            
            # Extract element region
            element_region = img_array[y1:y2, x1:x2]
            
            # Calculate visual properties
            visual_properties = {
                # Size relative to screen area
                "size": ((x2-x1) * (y2-y1)) / (ui_image.width * ui_image.height),
                
                # Contrast: difference between element and surroundings
                "contrast": self._calculate_contrast(img_array, x1, y1, x2, y2),
                
                # Color intensity: average saturation and value in HSV
                "color_intensity": self._calculate_color_intensity(element_region),
                
                # Motion: placeholder for animated elements (would need temporal data)
                "is_animated": 0.0,
                
                # Isolation: measure of whitespace around element
                "whitespace": self._calculate_isolation(img_array, x1, y1, x2, y2)
            }
            
            # Create UIElement
            ui_element = UIElement(
                id=f"element_{idx}",
                element_type=element['type'],
                bounds=element['bounds'],
                visual_properties=visual_properties,
                platform_specific={
                    "is_platform_pattern": False  # Could be determined based on type/position
                }
            )
            
            # Create AttentionCandidate
            candidate = AttentionCandidate(
                position=((x1 + x2) / (2 * ui_image.width), (y1 + y2) / (2 * ui_image.height)),
                candidate_type='ui_element',
                element_id=ui_element.id,
                element=ui_element
            )
            
            candidates.append(candidate)
        
        return candidates


    def _calculate_contrast(self, img_array, x1, y1, x2, y2) -> float:
        # Convert to LAB color space for better perceptual contrast
        if len(img_array.shape) == 3:
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        else:
            img_lab = cv2.cvtColor(cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB)

        # Get element region
        element = img_lab[y1:y2, x1:x2]
        
        # Calculate dynamic margin based on element size
        element_width = x2 - x1
        element_height = y2 - y1
        margin = min(20, int(max(element_width, element_height) * 0.2))
        
        # Define surrounding region boundaries
        y_min, y_max = max(0, y1-margin), min(img_lab.shape[0], y2+margin)
        x_min, x_max = max(0, x1-margin), min(img_lab.shape[1], x2+margin)
        
        # Calculate contrasts using multiple metrics
        contrasts = []
        
        # Luminance contrast (L channel)
        element_l = np.mean(element[:,:,0])
        surrounding_l = np.mean(img_lab[y_min:y_max, x_min:x_max, 0])
        luminance_contrast = abs(element_l - surrounding_l) / 100.0
        
        # Color contrast (a and b channels)
        element_a = np.mean(element[:,:,1])
        element_b = np.mean(element[:,:,2])
        surrounding_a = np.mean(img_lab[y_min:y_max, x_min:x_max, 1])
        surrounding_b = np.mean(img_lab[y_min:y_max, x_min:x_max, 2])
        color_contrast = np.sqrt((element_a - surrounding_a)**2 + (element_b - surrounding_b)**2) / 255.0
        
        # Combine contrasts with weights
        final_contrast = 0.7 * luminance_contrast + 0.3 * color_contrast
        
        return float(final_contrast)


    def _calculate_color_intensity(self, region: np.ndarray) -> float:
        """Calculate color intensity from RGB region"""
        if region.size == 0:
            return 0.0
            
        # Convert to HSV
        region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        
        # Calculate average saturation and value
        saturation = np.mean(region_hsv[:, :, 1]) / 255.0
        value = np.mean(region_hsv[:, :, 2]) / 255.0
        
        # Combine saturation and value
        return float((saturation + value) / 2)


    def _calculate_isolation(self, img_array: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate isolation based on surrounding whitespace"""
        # Define margin to check for whitespace
        margin = int(min(img_array.shape[0], img_array.shape[1]) * 0.05)
        
        # Calculate boundaries for surrounding region
        y_min, y_max = max(0, y1-margin), min(img_array.shape[0], y2+margin)
        x_min, x_max = max(0, x1-margin), min(img_array.shape[1], x2+margin)
        
        # Get surrounding region
        surround = img_array[y_min:y_max, x_min:x_max]
        
        if surround.size == 0:
            return 0.0
            
        # Convert to grayscale if needed
        if len(surround.shape) == 3:
            surround = np.mean(surround, axis=2)
            
        # Calculate whitespace as ratio of light pixels
        whitespace_ratio = np.mean(surround > 240) # Assuming 240+ is "white"
        return float(whitespace_ratio)


    def predict_attention(self,
                         platform: Platform,
                         task: str,
                         tech_savv: int,
                         ui_image: Image,
                         elements_data: List[Dict]) -> Dict[str, any]:
        """
        Predict attention points for a UI screenshot.
        
        Args:
            ui_image: PIL Image of the UI screenshot
            task: Description of the user's current task
            elements_data: List of dictionaries containing:
                - type: str (button, text, icon, etc)
                - text: str (if any)
                - bounds: Dict with x1,y1,x2,y2 in normalized coordinates
            
        Returns:
            Dictionary containing primary focus, secondary focuses, and attention distribution
        """
        self.platform = platform
        self.tech_savv = tech_savv
        
        # Extract UI elements from image with provided data
        attention_candidates = self._extract_ui_elements_from_image(ui_image, elements_data)
        
        # Add platform hotspots
        attention_candidates.extend(self._generate_platform_hotspots())
        
        # First, merge overlapping candidates
        # merged_candidates = self._merge_overlapping_candidates(attention_candidates)
        merged_candidates = attention_candidates
        attention_points = []
        
        for candidate in merged_candidates:
            # Base scores for all candidates
            position_score = self._calculate_base_position_score(candidate.position)
            task_score = self._calculate_base_task_score(candidate.position, task)
            visual_score = self._calculate_visual_attraction_score(candidate)
            
            # Weight the scores based on tech savviness
            # Tech savvy users: position & task matter more
            # Non-tech savvy users: visual attraction matters more
            if tech_savv >= 7:
                final_score = (
                    position_score * 0.27 +
                    task_score * 0.55 +
                    visual_score * 0.18
                )
            elif tech_savv <= 3:
                final_score = (
                    position_score * 0.22 +
                    task_score * 0.23 +
                    visual_score * 0.55
                )
            else:  # Medium tech savviness
                final_score = (
                    position_score * 0.33 +
                    task_score * 0.33 +
                    visual_score * 0.34
                )
            
            attention_points.append({
                "element_id": candidate.element_id,
                "position": candidate.position,
                "candidate_type": candidate.candidate_type,
                "score": final_score,
                "component_scores": {
                    "position": position_score,
                    "task": task_score,
                    "visual": visual_score
                }
            })
        
        # Sort by score
        attention_points.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top_k points and normalize their scores to get confidence distribution
        top_points = attention_points
        scores = np.array([point["score"] for point in top_points])
        
        # Add confidence to top points
        for point, score in zip(top_points, scores):
            point["score"] = float(score)
            point["reasoning"] = self._generate_reasoning_v2(
                next(c for c in merged_candidates if c.position == point["position"]),
                point["component_scores"]["position"],
                point["component_scores"]["task"],
                point["component_scores"]["visual"]
            )
        
        # Merge close points
        top_points = self._merge_close_points(top_points)
        
        return {
            "primary_focus": top_points[0] if top_points else None,
            "secondary_focuses": top_points[1:],
            "attention_distribution": top_points
        }


    def _generate_reasoning_v2(self,
                             candidate: AttentionCandidate,
                             position_score: float,
                             task_score: float,
                             visual_score: float) -> str:
        """Generate improved reasoning for the prediction"""
        reasons = []
        
        if position_score > 0.7:
            reasons.append(f"Strong platform convention for {self.platform.value}")
        if task_score > 0.7:
            reasons.append("Highly relevant position for task")
        if visual_score > 0.7:
            reasons.append("Visually prominent")
        
        if candidate.candidate_type == 'ui_element':
            # Add UI element specific reasoning
            if candidate.element.visual_properties.get("contrast", 0) > 0.7:
                reasons.append("High visual contrast")
            if candidate.element.platform_specific.get("is_platform_pattern", False):
                reasons.append("Follows platform pattern")
        elif candidate.candidate_type == 'platform_hotspot':
            # Add hotspot-specific reasoning
            if task_score > 0.7:
                reasons.append("Task-relevant hotspot")
            if self.tech_savv >= 7:
                reasons.append("Tech-savvy user preference")
        
        return ", ".join(reasons) if reasons else "Based on general UI principles"


    def _generate_platform_hotspots(self) -> List[AttentionCandidate]:
        """Generate platform hotspots based on the hotspot definitions"""
        hotspots = []
        platform_hotspots = self.hotspot_definitions[self.platform]
        
        for hotspot_name, hotspot_data in platform_hotspots.items():
            # Calculate center point from bounds
            bounds = hotspot_data["bounds"]
            center_x = (bounds["x1"] + bounds["x2"]) / 2
            center_y = (bounds["y1"] + bounds["y2"]) / 2
            
            # Create hotspot candidate
            hotspot = AttentionCandidate(
                position=(center_x, center_y),
                candidate_type='platform_hotspot',
                element_id=f"hotspot_{hotspot_name}"
            )
            hotspots.append(hotspot)
        
        return hotspots


    def visualize_attention(self, 
                           attention_result: Dict[str, any],
                           ui_image: Image,
                           alpha: float = 0.6,
                           top_k: int = 5) -> Image:
        """
        Visualize top attention points overlaid on the UI screenshot.
        Red circles for UI elements, green circles for hotspots.
        Coordinates are normalized from bottom-left origin.
        
        Args:
            attention_result: Result from predict_attention
            ui_image: PIL Image of the UI screenshot
            alpha: Transparency of the overlay (0-1)
        
        Returns:
            PIL Image with attention visualization overlaid
        """
        from PIL import Image, ImageDraw
        
        # Convert to RGBA if not already
        ui_image = ui_image.convert('RGBA')
        
        # Create a transparent overlay for the heatmap
        overlay = Image.new('RGBA', ui_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Get image dimensions for coordinate conversion
        width, height = ui_image.size
        
        # Function to convert normalized coordinates to pixel coordinates
        # Flip y-coordinate since PIL uses top-left origin
        def norm_to_pixel(x: float, y: float) -> Tuple[int, int]:
            return (int(x * width), int(y * height))  # Flip y coordinate
        
        # Get the top attention points (these already have normalized confidence scores)
        points = [attention_result["primary_focus"]] + attention_result["secondary_focuses"]
        points = points[:top_k]
        scores = np.array([point["score"] for point in points])
        
        # Apply softmax to get confidence distribution
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        confidences = exp_scores / exp_scores.sum()
        
        # Draw attention circles
        for i, point in enumerate(points):
            x, y = point["position"]
            confidence = confidences[i]  # Already normalized by softmax
            
            # Convert to pixel coordinates (y is now flipped)
            px, py = norm_to_pixel(x, y)
            
            # Calculate radius based on image size (e.g., 5% of width)
            radius = int(width * 0.05)
            
            # Color intensity based only on confidence
            intensity = int(255 * confidence * alpha)
            
            # Choose color based on whether it's a hotspot or UI element
            color = (0, 255, 0, intensity) if point["candidate_type"] == "platform_hotspot" else (255, 0, 0, intensity)  # Green for hotspots, Red for UI elements
            
            # Single solid circle for each attention point
            draw.ellipse(
                [(px - radius, py - radius), (px + radius, py + radius)],
                fill=color,
                outline=None
            )
        
        # Blend the overlay with the original image
        result = Image.alpha_composite(ui_image, overlay)
        
        # Add small confidence labels
        draw = ImageDraw.Draw(result)
        for i, point in enumerate(points):
            x, y = point["position"]
            px, py = norm_to_pixel(x, y)  # y is already flipped
            confidence = confidences[i]
            
            # Draw white label with confidence percentage
            label = f'{confidence*100:.0f}%'
            draw.text(
                (px + 10, py + 10),
                label,
                fill=(255, 255, 255, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=2
            )
        
        return result


    def _merge_close_points(self, attention_points: List[Dict]) -> List[Dict]:
        """
        Merge attention points that are close to each other, prioritizing UI elements.
        Each point can merge with at most 2 nearest neighbors, using diminishing returns
        for score combination.
        """
        def sort_key(point):
            return (point["candidate_type"] == "ui_element", point["score"])
        
        sorted_points = sorted(attention_points, key=sort_key, reverse=True)
        merged_points = []
        processed_ids = set()  # Track processed points by element_id
        distance_threshold = self.distance_threshold.get(self.platform, 0.05)
        
        for i, point in enumerate(sorted_points):
            # Skip if this point's id has already been processed
            point_id = point.get("element_id", f"hotspot_{i}")  # Fallback for hotspots
            if point_id in processed_ids:
                continue
            
            # Get current point position
            x1, y1 = point["position"]
            base_score = point["score"]
            
            # Find all close points and their distances
            nearby_points = []
            for j, other_point in enumerate(sorted_points[i+1:]):
                other_id = other_point.get("element_id", f"hotspot_{i+1+j}")
                if other_id in processed_ids:
                    continue
                
                x2, y2 = other_point["position"]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if distance <= distance_threshold:
                    nearby_points.append((distance, other_point, other_id))
            
            # Sort by distance and take at most 2 nearest points
            nearby_points.sort(key=lambda x: x[0])  # Sort by distance
            close_points = nearby_points[:2]  # Take at most 2 nearest points
            
            # Calculate merged score with diminishing returns
            if close_points:
                # Sort all scores (including base_score) in descending order
                all_scores = sorted([base_score] + [p[1]["score"] for p in close_points], reverse=True)
                merged_score = all_scores[0]  # Start with highest score
                
                # Add diminishing contributions from other scores
                for idx, score in enumerate(all_scores[1:]):
                    # Each additional score contributes less
                    # 2nd score adds 10%, 3rd adds 5%
                    contribution = score * (0.1 / (2 ** idx))
                    merged_score += contribution
                
                # Mark all merged points as processed
                for _, _, other_id in close_points:
                    processed_ids.add(other_id)
            else:
                merged_score = base_score
            
            # Create merged point
            merged_point = point.copy()
            merged_point["score"] = merged_score
            
            # Update reasoning if points were merged
            if close_points:
                merged_point["reasoning"] += f" (Combined with {len(close_points)} nearby points)"
            
            # Mark current point as processed
            processed_ids.add(point_id)
            merged_points.append(merged_point)
        
        # Sort by score
        merged_points = sorted(merged_points, key=lambda x: x["score"], reverse=True)
        
        # Assert no duplicate element_ids (excluding None values)
        element_ids = [p["element_id"] for p in merged_points if p["element_id"] is not None]
        assert len(element_ids) == len(set(element_ids)), "Duplicate element_ids found in merged points"
        
        # Assert no duplicate positions
        positions = [p["position"] for p in merged_points]
        assert len(positions) == len(set(map(tuple, positions))), "Duplicate positions found in merged points"
        
        return merged_points


# Example usage:
"""
# Initialize predictor
predictor = UIAttentionPredictor(
    platform=Platform.ANDROID,
    tech_savviness=3
)

# Example elements data
elements_data = [
    {
        "type": "button",
        "text": "Settings",
        "bounds": {
            "x1": 0.1,  # These are normalized coordinates (0-1)
            "y1": 0.1,
            "x2": 0.2,
            "y2": 0.2
        }
    },
    {
        "type": "icon",
        "text": "menu",
        "bounds": {
            "x1": 0.8,
            "y1": 0.1,
            "x2": 0.9,
            "y2": 0.2
        }
    }
]

# Get prediction
result = predictor.predict_attention(
    ui_image=Image.open("path_to_ui_image.png"),
    task="Find the settings menu",
    elements_data=elements_data
)

print(result)
""" 