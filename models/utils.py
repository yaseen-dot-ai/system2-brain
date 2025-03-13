from PIL import Image, ImageDraw
import numpy as np
import torch
import matplotlib.pyplot as plt

import io
import base64
import traceback
import json

from openai import OpenAI
from langchain_anthropic import ChatAnthropic

from pydantic import BaseModel
from typing import Optional, List

import os


def get_cropped_icon(image, element, ratio=0.30):
    highlighted_image = draw_bounding_box(image, element)
    image_width, image_height = highlighted_image.size
    x, y = element["position"]
        
    # Convert normalized coordinates to pixel coordinates
    pixel_x = int(x * image_width)
    pixel_y = int(y * image_height)
    
    box_size = int(ratio * image_width)
    
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
    cropped_icon = Image.fromarray(cropped_array).convert('RGB')
    return cropped_icon

def process_single_image(image, model_processor, prompt=None):
    """Process a single image through the model to generate caption/description.
    
    Args:
        image: Single PIL image
        model_processor: Dict containing model and processor
        prompt: Optional prompt text to guide generation
    """
    model, processor = model_processor['model'], model_processor['processor']
    
    # Set default prompt based on model type
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"
    
    device = model.device
    
    # Process single image
    if model.device.type == 'cuda':
        inputs = processor(
            images=image, 
            text=prompt, 
            return_tensors="pt", 
            do_resize=False
        ).to(device=device, dtype=torch.float16)
    else:
        inputs = processor(
            images=image, 
            text=prompt, 
            return_tensors="pt"
        ).to(device=device)
    
    # Generate text based on model type
    if 'florence' in model.config.name_or_path:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=20,
            num_beams=1,
            do_sample=False
        )
    else:
        generated_ids = model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_return_sequences=1
        )
    
    # Decode and clean up generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

def evaluate_cropped_icon(cropped_icon, task_description, cross_encoder, tokenizer):
    cropped_icon.resize((256, 256), resample=Image.Resampling.LANCZOS)
    caption = process_single_image(cropped_icon, "What is the purpose of the highlighted UI element in this image ? Answer in one sentence.")
    
    cross_encoder.eval()
    with torch.no_grad():
        inputs = tokenizer([task_description, caption], padding=True, truncation=True, return_tensors='pt', max_length=512)
        score = cross_encoder(**inputs, return_dict=True).logits.view(-1, ).float()[0]
    
    return score


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


async def chat_anthropic(prompt: str, image: Optional[Image.Image]=None, history_msgs: List[dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant."}], model_name: str="claude-3-5-sonnet-20240620", output_schema: Optional[BaseModel]=None) -> str:
    model = ChatAnthropic(model=model_name, temperature=0, max_tokens=8000, api_key=os.getenv("ANTHROPIC_API_KEY"))
    messages = history_msgs + [{"role": "user", "content": prompt}]
    if image:
        # Encode image to base64
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        messages = history_msgs + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + image_base64}
                    }
                ]
            }
        ]
    if output_schema:
        response = await model.with_structured_output(output_schema).ainvoke(messages)
    else:
        response = await model.ainvoke(messages)
    return response.content


def attention_decay(index, total_elements):
    """
    Implements a more realistic attention decay function.
    - Sharp initial drop (first few elements get significantly more attention)
    - Followed by a more gradual decline
    - Maintains a minimum attention level (people don't completely ignore elements)
    """
    # Parameters to tune the attention curve
    initial_drop_rate = 2.5  # Controls how sharp the initial attention drop is
    base_attention = 0.2    # Minimum attention level (never goes to zero)
    
    # Normalized position (0 to 1)
    normalized_pos = index / total_elements
    
    # Modified exponential decay with baseline
    score = (1 - base_attention) * np.exp(-initial_drop_rate * normalized_pos) + base_attention
    return score


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


def f_scan_pattern_scores(elements_ref, last_element, debug):
    spatial_grid = create_spatial_grid(elements_ref)
    scored_elements = []
    
    # Check if last_element exists in the grid
    last_element_found = False
    if last_element:
        for row in spatial_grid:
            for element in row:
                if element["element_id"] == last_element["element_id"]:
                    last_element_found = True
                    break
            if last_element_found:
                break
    
    # If last_element was found, use layered scanning from that point
    if last_element_found:
        # First collect elements row by row (layered pattern)
        for i in range(len(spatial_grid)):
            row = spatial_grid[i]
            for j in range(len(row)):
                scored_elements.append(row[j])
                
        # Remove elements up to and including last_element
        for index, element in enumerate(scored_elements):
            if element["element_id"] == last_element["element_id"]:
                scored_elements = scored_elements[index + 1:]
                break
                
        # Apply simple attention decay to remaining elements
        total_elements = len(scored_elements)
        for index, element in enumerate(scored_elements):
            score = attention_decay(index, total_elements)
            scored_elements[index] = (element, score)
            
    # If starting from beginning or last_element not found, use F-pattern
    else:
        # Collect and score elements in F-pattern
        for i in range(len(spatial_grid)):
            row = spatial_grid[i]
            for j, element in enumerate(row):
                base_score = attention_decay(len(scored_elements), len(elements_ref))
                
                # Apply F-pattern boost only for initial scan
                if i < 2:  # First two rows get full scan attention
                    boost = 1.0
                elif j < 2:  # Left side gets attention after first two rows
                    boost = 0.8
                else:  # Other elements get base attention
                    boost = 0.5
                    
                scored_elements.append((element, min(1.0, base_score * boost)))
    
    return scored_elements


def z_scan_pattern_scores(elements_ref, last_element, debug):
    spatial_grid = create_spatial_grid(elements_ref)
    scored_elements = []
    
    # Traverse the spatial grid in Z-pattern
    for i in range(len(spatial_grid)):
        row = spatial_grid[i]
        if i % 2 == 0:  # Moving right
            for j in range(len(row)):
                scored_elements.append(row[j])
        else:  # Moving left
            for j in range(len(row) - 1, -1, -1):
                scored_elements.append(row[j])
    
    # Remove elements up to and including last_element
    if last_element:
        for index, element in enumerate(scored_elements):
            if element["element_id"] == last_element["element_id"]:
                scored_elements = scored_elements[index + 1:]
                break
    
    # Score the elements using the attention decay function
    total_elements = len(scored_elements)
    for index, element in enumerate(scored_elements):
        score = attention_decay(index, total_elements)
        scored_elements[index] = (element, score)
    
    return scored_elements


def layered_scan_pattern_scores(elements_ref, last_element, debug, layer_size=2):
    spatial_grid = create_spatial_grid(elements_ref)
    scored_elements = []
    
    # Traverse the spatial grid in layered pattern
    for layer_start in range(0, len(spatial_grid), layer_size):
        layer_end = min(layer_start + layer_size, len(spatial_grid))
        
        # Process each row within the current layer
        for i in range(layer_start, layer_end):
            row = spatial_grid[i]
            for j in range(len(row)):
                scored_elements.append(row[j])
    
    # Remove elements up to and including last_element
    if last_element:
        for index, element in enumerate(scored_elements):
            if element["element_id"] == last_element["element_id"]:
                scored_elements = scored_elements[index + 1:]
                break
    
    # Score the elements using the attention decay function
    total_elements = len(scored_elements)
    for index, element in enumerate(scored_elements):
        score = attention_decay(index, total_elements)
        scored_elements[index] = (element, score)
    
    return scored_elements


def spotted_pattern_scores(elements_ref, last_element, debug):
    # Sort elements by their visual score in descending order
    visual_elements = sorted(elements_ref, key=lambda x: x["visual_score"], reverse=True)
    scored_elements = []
    
    # Remove elements up to and including last_element
    if last_element:
        for index, element in enumerate(visual_elements):
            if element["element_id"] == last_element["element_id"]:
                visual_elements = visual_elements[index + 1:]
                break
    
    # Score remaining elements using attention decay
    total_elements = len(visual_elements)
    for index, element in enumerate(visual_elements):
        score = attention_decay(index, total_elements)
        scored_elements.append((element, score))
    
    return scored_elements


def find_scanning_pattern_scores(elements_ref, pattern, last_element, debug=False):
    if pattern == "Spotted Pattern":
        return spotted_pattern_scores(elements_ref, last_element, debug)
    elif pattern == "Z-Pattern":
        return z_scan_pattern_scores(elements_ref, last_element, debug)
    elif pattern == "F-Pattern":
        return f_scan_pattern_scores(elements_ref, last_element, debug)
    elif pattern == "Layered Pattern":
        return layered_scan_pattern_scores(elements_ref, last_element, debug)
    
    return None


def draw_bounding_box(image, element):
    draw = ImageDraw.Draw(image)
    draw.rectangle(element["bounds"], outline="red", width=2)
    return image