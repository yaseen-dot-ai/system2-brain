from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from models.ui_attention_predictor import Platform
import matplotlib.pyplot as plt
import io
import base64
import traceback
import json
from openai import OpenAI

cos = nn.CosineSimilarity(dim=0, eps=1e-8)

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess the input image for model inference
    """
    # Add your image preprocessing logic here
    # Example:
    # - Resize
    # - Normalize
    # - Convert to tensor
    pass


def postprocess_output(model_output: torch.Tensor) -> list:
    """
    Convert model output to the desired format
    """
    # Add your output processing logic here
    pass


def validate_inputs(age: int, platform: str, task: str, tech_saviness: int) -> bool:
    """
    Validate all input parameters
    """
    try:
        # Add validation logic
        assert isinstance(age, int) and 0 <= age <= 120
        assert isinstance(platform, Platform)  # platform should already be a Platform enum
        assert isinstance(tech_saviness, int) and 1 <= tech_saviness <= 10
        return True
    except:
        return False
    

def evaluate_cropped_icon(model, processor, embed_model, cropped_icon, task_description, threshold=0.4):
    # cropped_icon = cropped_icon.convert('RGB')
    # cropped_icon.resize((64, 64), resample=Image.Resampling.LANCZOS)
    
    caption = chat_vllm(cropped_icon, "What are the UI elements in this image and what are they used for ? Answer in one sentence.")
    
    embeddings = embed_model.encode([caption, task_description], convert_to_tensor=True)
    print(f"caption: {caption}, task_description: {task_description}")
    similarity = cos(embeddings[0], embeddings[1])
    print(f"similarity: {similarity.item()}")
    return similarity.item() > threshold


def get_color_for_timestep(timestep, max_timesteps):
    # Define key colors in RGB
    colors = [
        (255, 0, 0),      # Red
        (255, 165, 0),    # Orange
        (255, 255, 0),    # Yellow
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (128, 0, 128),    # Purple
        (139, 69, 19),    # Brown
        (0, 0, 0)         # Black
    ]
    
    # Calculate which color pair we're between
    num_transitions = len(colors) - 1
    section_size = max_timesteps / num_transitions
    
    # Find current section and progress within that section
    section = int(timestep / section_size)
    section = min(section, num_transitions - 1)  # Clamp to avoid overflow
    
    progress = (timestep % section_size) / section_size
    
    # Get the two colors to interpolate between
    color1 = colors[section]
    color2 = colors[section + 1]
    
    # Interpolate between the two colors
    r = int(color1[0] + (color2[0] - color1[0]) * progress)
    g = int(color1[1] + (color2[1] - color1[1]) * progress)
    b = int(color1[2] + (color2[2] - color1[2]) * progress)
    
    return (r, g, b, int(255 * 0.8))  # Keep alpha at 0.8

def draw_attention(attention_point, ui_image, timestep_count, total_timesteps=100) -> tuple[Image.Image, tuple[int, int, int, int]]:
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
    def norm_to_pixel(x: float, y: float):
        return (int(x * width), int(y * height))  # Flip y coordinate
    
    x, y = attention_point["position"]
    
    # Convert to pixel coordinates (y is now flipped)
    px, py = norm_to_pixel(x, y)
    
    # Calculate radius based on image size (e.g., 10% of width)
    radius = int(width * 0.05)
    
    color = get_color_for_timestep(timestep_count, total_timesteps)  # You'll need to pass these parameters
    
    # Single solid circle for each attention point
    draw.ellipse(
        [(px - radius, py - radius), (px + radius, py + radius)],
        fill=color,
        outline=None
    )
    
    # Add this line to blend the overlay with the original image
    ui_image = Image.alpha_composite(ui_image, overlay)
    return ui_image, color


def create_spatial_grid(elements):
    # Sort elements by y-coordinate first (rows)
    sorted_by_y = sorted(elements, key=lambda x: x["position"][1])
    
    # Group elements into rows based on y-coordinate proximity
    rows = []
    current_row = []
    y_threshold = 0.05  # Adjust based on your UI layout
    
    for element in sorted_by_y:
        if not current_row or abs(element["position"][1] - current_row[0]["position"][1]) < y_threshold:
            current_row.append(element)
        else:
            current_row.sort(key=lambda x: x["position"][0])
            rows.append(current_row)
            current_row = [element]
    
    if current_row:
        current_row.sort(key=lambda x: x["position"][0])
        rows.append(current_row)
    
    return rows


def z_scan_pattern(elements_ref, last_element, debug):
    if not last_element:
        # Start from top-left
        if debug:
            print("Starting from top-left element: 0, 0")
        return min(elements_ref, key=lambda x: (x["position"][1], x["position"][0]))
    
    spatial_grid = create_spatial_grid(elements_ref)
    current_row_idx = None
    current_col_idx = None
    
    # Find current position in grid
    for i, row in enumerate(spatial_grid):
        for j, element in enumerate(row):
            if element["element_id"] == last_element["element_id"]:
                current_row_idx = i
                current_col_idx = j
                if debug:
                    print(f"found last element in grid: {current_row_idx}, {current_col_idx}")
                break
        if current_row_idx is not None:
            break
    
    if current_row_idx is None:
        if debug:
            print("current_row_idx is None")
        return None
    
    if debug:
        print(f"moving in z-pattern: {current_row_idx}, {current_col_idx} ->", end=" ")
    # Z-pattern movement
    if current_row_idx % 2 == 0:  # Moving right
        if current_col_idx < len(spatial_grid[current_row_idx]) - 1:
            if debug:
                print(f"{current_row_idx}, {current_col_idx + 1}")
            return spatial_grid[current_row_idx][current_col_idx + 1]
        elif current_row_idx < len(spatial_grid) - 1:
            # Move to next row, starting from right
            if debug:
                print(f"{current_row_idx + 1}, {len(spatial_grid[current_row_idx + 1]) - 1}")
            return spatial_grid[current_row_idx + 1][-1]
    else:  # Moving left
        if current_col_idx > 0:
            if debug:
                print(f"{current_row_idx}, {current_col_idx - 1}")
            return spatial_grid[current_row_idx][current_col_idx - 1]
        elif current_row_idx < len(spatial_grid) - 1:
            # Move to next row, starting from left
            if debug:
                print(f"{current_row_idx + 1}, 0")
            return spatial_grid[current_row_idx + 1][0]
    
    return None


def find_current_position(last_element, spatial_grid):
    """Helper function to find element position in grid"""
    for i, row in enumerate(spatial_grid):
        for j, element in enumerate(row):
            if element["element_id"] == last_element["element_id"]:
                return i, j
    return None, None


def f_scan_pattern(elements_ref, last_element, debug):
    import random
    
    if not last_element:
        if debug:
            print("Starting from top-left element: 0, 0")
        return min(elements_ref, key=lambda x: (x["position"][1], x["position"][0]))
    
    spatial_grid = create_spatial_grid(elements_ref)
    current_row_idx, current_col_idx = find_current_position(last_element, spatial_grid)
    
    if current_row_idx is None:
        if debug:
            print("Element not found in grid")
        return None    
    
    if debug:
        print(f"Moving in f-pattern: {current_row_idx}, {current_col_idx} ->", end=" ")
        
    # If not in first column, chance to return to first columns
    if current_col_idx > 0:
        cols_in_row = len(spatial_grid[current_row_idx])
        jump_probability = 2/(cols_in_row + 3)        
        if random.random() < jump_probability:
            target_col = random.randint(0, 1)
            if target_col < cols_in_row:
                if debug:
                    print(f"{current_row_idx}, {target_col}")
                return spatial_grid[current_row_idx][target_col]
    
    # If in first two columns of lower rows, chance to jump to top rows
    if current_row_idx > 1 and current_col_idx < 2:
        jump_probability = 2/(len(spatial_grid) + 4)
        
        if random.random() < jump_probability:
            target_row = random.randint(0, 1)
            if current_col_idx < len(spatial_grid[target_row]):
                if debug:
                    print(f"{target_row}, {current_col_idx}")
                return spatial_grid[target_row][current_col_idx]
    
    # Default sequential movement
    if current_col_idx < len(spatial_grid[current_row_idx]) - 1:
        if debug:
            print(f"{current_row_idx}, {current_col_idx + 1}")
        return spatial_grid[current_row_idx][current_col_idx + 1]
    elif current_row_idx < len(spatial_grid) - 1:
        if debug:
            print(f"{current_row_idx + 1}, 0")
        return spatial_grid[current_row_idx + 1][0]
    
    if debug:
        print("No valid moves remaining")
    return None


def layered_scan_pattern(elements_ref, last_element, debug, layer_size=2):
    if not last_element:
        # Start from top-left
        return min(elements_ref, key=lambda x: (x["position"][1], x["position"][0]))
    
    spatial_grid = create_spatial_grid(elements_ref)
    current_row_idx = None
    current_col_idx = None
    
    # Find current position in grid
    for i, row in enumerate(spatial_grid):
        for j, element in enumerate(row):
            if element["element_id"] == last_element["element_id"]:
                current_row_idx = i
                current_col_idx = j
                if debug:
                    print(f"found last element in grid: {current_row_idx}, {current_col_idx}")
                break
        if current_row_idx is not None:
            break
    
    if current_row_idx is None:
        if debug:
            print("current_row_idx is None")
        return None
    
    if debug:
        print(f"Moving in layered pattern: {current_row_idx}, {current_col_idx} ->", end=" ")
    
    # Determine which layer we're in
    current_layer = current_row_idx // layer_size
    layer_start = current_layer * layer_size
    layer_end = min(layer_start + layer_size, len(spatial_grid))
    
    # Scan within current layer
    if current_col_idx < len(spatial_grid[current_row_idx]) - 1:
        if debug:
            print(f"{current_row_idx}, {current_col_idx + 1}")
        return spatial_grid[current_row_idx][current_col_idx + 1]
    elif current_row_idx < layer_end - 1:
        if debug:
            print(f"{current_row_idx + 1}, 0")
        return spatial_grid[current_row_idx + 1][0]
    elif layer_end < len(spatial_grid):
        if debug:
            print(f"{layer_end}, 0")
        return spatial_grid[layer_end][0]
    
    if debug:
        print("Reached end of grid!")
    return None

def spotted_pattern(elements_ref, last_element, debug):
    visual_elements = sorted(elements_ref, key=lambda x: x["component_scores"]["visual"], reverse=True)
    if not last_element:
        return visual_elements[0]
    for i, element in enumerate(visual_elements):
        if element["element_id"] == last_element["element_id"] and i < len(visual_elements) - 1:
            return visual_elements[i+1]
    return None

def find_next_element_scan(elements_ref, pattern, last_element, debug=False):
    if pattern == "Spotted Pattern":
        return spotted_pattern(elements_ref, last_element, debug)
    elif pattern == "Z-Pattern":
        return z_scan_pattern(elements_ref, last_element, debug)
    elif pattern == "F-Pattern":
        return f_scan_pattern(elements_ref, last_element, debug)
    elif pattern == "Layered Pattern":
        return layered_scan_pattern(elements_ref, last_element, debug)
    
    return None

def display_image(image: Image.Image, jup=True):
    if jup:
        image.show()
        return
    
    # Convert PIL Image to matplotlib format and display
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show(block=True)  # This will block until the window is closed
    plt.close()  # Explicitly close the figure

def chat_vllm(image: Image.Image, prompt: str, base_url="https://uitars.wisit.io", api_key: str="dummy-key", model_name: str="bytedance-research/UI-TARS-2B-SFT") -> str:
    # Encode image to base64
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50,
            temperature=0
        )
        
        if response.status != 200:
            print("\nFull API Request Details:")
            print("URL:", f"{base_url}/v1/chat/completions")
            print("Auth:", f"Bearer {api_key}")
            print("Messages:", json.dumps(response.request.json, indent=2))
            print("\nResponse Status:", response.status)
            print("Response Body:", response.text)
            raise Exception(f"API request failed with status {response.status}: {response.text}")
            
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in chat_vllm, traceback: {traceback.format_exc()}")
        return ""