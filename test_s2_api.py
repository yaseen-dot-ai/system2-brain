import base64
import requests
import json
from PIL import Image

url = "http://localhost:8000/s2"

image = Image.open("dashboard.png")
image_base64 = base64.b64encode(image.tobytes()).decode('utf-8')

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
  "bboxes": [
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
      "height": 51.71875,
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

print(response.text)