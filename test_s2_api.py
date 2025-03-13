import base64
import requests
import json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

url = "https://5591-14-143-179-90.ngrok-free.app/s2"

image = Image.open("dashboard.png")
buffered = BytesIO()
image.save(buffered, format="PNG")
image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

payload = json.dumps({
  "task": "add a widget to the dashboard",
  "screenshot": image_base64,
  "context": {
    "tech_savviness": "MEDIUM",
    "domain_familiarity": "EXPERT",
    "domain": "devops engineers",
    "role": "devops",
    "gender": "male",
    "age": "25"
  },
  "scratchpad": [],
  "prev_state": {
    "action": "click",
    "bounding": {
      "x": 51,
      "y": 17,
      "width": 24,
      "height": 24,
      "text": "element element element element element icon",
      "type": "svg",
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(1) > button > span:nth-child(1) > svg"
    }
  },
  "elements": [
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(1) > button > span:nth-child(1) > svg",
      "text": "element element element element element icon",
      "type": "svg",
      "width": 24,
      "height": 24,
      "x": 51,
      "y": 17
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(2) > div > div > ul:nth-child(2) > a:nth-child(1) > div > div:nth-child(1) > img",
      "text": "",
      "type": "img",
      "width": 20,
      "height": 20,
      "x": 21,
      "y": 106
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(2) > div > div > ul:nth-child(2) > a:nth-child(2) > div > div:nth-child(1) > img",
      "text": "",
      "type": "img",
      "width": 20,
      "height": 20,
      "x": 21,
      "y": 160
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(2) > div > div > ul:nth-child(2) > a:nth-child(3) > div > div:nth-child(1) > img",
      "text": "",
      "type": "img",
      "width": 24,
      "height": 24,
      "x": 19,
      "y": 212
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(2) > div > div > ul:nth-child(2) > a:nth-child(4) > div > div:nth-child(1) > img",
      "text": "",
      "type": "img",
      "width": 20,
      "height": 20,
      "x": 21,
      "y": 268
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(2) > div > div > ul:nth-child(2) > a:nth-child(5) > div > div:nth-child(1) > img",
      "text": "",
      "type": "img",
      "width": 20,
      "height": 20,
      "x": 21,
      "y": 322
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(2) > div > div > ul:nth-child(2) > a:nth-child(6) > div > div:nth-child(1) > img",
      "text": "",
      "type": "img",
      "width": 20,
      "height": 20,
      "x": 21,
      "y": 376
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(2) > div > div > ul:nth-child(2) > a:nth-child(7) > div > div:nth-child(1) > img",
      "text": "",
      "type": "img",
      "width": 20,
      "height": 20,
      "x": 21,
      "y": 430
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div:nth-child(2) > div > div > ul:nth-child(3) > div > div:nth-child(1)",
      "text": "S",
      "type": "div",
      "width": 36,
      "height": 36,
      "x": 13,
      "y": 633
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(2) > div > div:nth-child(1) > div > div > img:nth-child(1)",
      "text": "",
      "type": "img",
      "width": 20,
      "height": 16,
      "x": 86,
      "y": 26
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(2) > div > div:nth-child(1) > div > div > div:nth-child(2)",
      "text": "Go Back",
      "type": "div",
      "width": 24,
      "height": 52,
      "x": 108,
      "y": 24
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(2) > div > div:nth-child(4) > div:nth-child(1) > div > img:nth-child(1)",
      "text": "",
      "type": "img",
      "width": 50,
      "height": 50,
      "x": 93,
      "y": 157
    },
    {
      "selector": "html > body:nth-child(2) > div:nth-child(1) > div > div:nth-child(2) > div > div:nth-child(4) > div:nth-child(1) > div > div:nth-child(2) > div:nth-child(1)",
      "text": "Cloud Cost Analysis",
      "type": "div",
      "width": 24,
      "height": 374,
      "x": 157,
      "y": 152
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

if response.status_code == 200:
    result = response.json()
    print(result)
    
    # Create a copy of the image for drawing
    image_with_box = image.copy()
    draw = ImageDraw.Draw(image_with_box)
    
    # Get the selected element
    selected_element = result["selected_element"]
    action = result["action"]
    
    # Draw bounding box
    box_coords = [
        selected_element["x"],
        selected_element["y"],
        selected_element["x"] + selected_element["width"],
        selected_element["y"] + selected_element["height"]
    ]
    draw.rectangle(box_coords, outline="red", width=2)
    
    # Add action label
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw label background
    label_text = f"{action.upper()}: {selected_element['text'] or selected_element['type']}"
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Position label above the bounding box
    label_x = selected_element["x"]
    label_y = max(0, selected_element["y"] - text_height - 5)
    
    # Draw label background
    draw.rectangle(
        [label_x, label_y, label_x + text_width, label_y + text_height],
        fill="red"
    )
    
    # Draw label text
    draw.text((label_x, label_y), label_text, fill="white", font=font)
    
    # Convert PIL Image to numpy array for matplotlib
    img_array = np.array(image_with_box)
    
    # Display with matplotlib
    plt.figure(figsize=(15, 10))
    plt.imshow(img_array)
    plt.axis('off')  # Hide axes
    plt.show()
else:
    print(response.json())