# System 2 API Documentation

## Overview
The System 2 API provides an endpoint for simulating System 2 reasoning in web interaction decisions. It processes screenshots and context to determine the most appropriate element to interact with on a webpage.

## Base URL
```
http://localhost:8000
```

## Endpoints

### POST /s2
Makes a System 2 reasoning decision based on the provided screenshot and context.

#### Request Format
```json
{
    "task": string,                    // Required: The task that needs to be completed
    "screenshot": string,              // Required: Base64 encoded image or image URL
    "previous_screenshot": string,     // Optional: Previous page screenshot for change detection
    "context": object,                 // Required: User and domain context
    "scratchpad": array,               // Optional: List of previous actions and their content
    "elements": array,                 // Required: List of UI elements on the page
    "prev_action": object              // Optional: Previous action taken
}
```

##### Field Details:
- `task`: Description of what needs to be accomplished
- `screenshot`: Can be either:
  - Base64 encoded image (with or without data URL prefix)
  - Direct URL to an image
- `previous_screenshot`: Previous page screenshot (same format as screenshot)
- `context`: JSON object containing:
  - `age`: User's age (integer)
  - `domain_familiarity`: User's familiarity with the domain ("beginner", "intermediate", "expert")
  - `domain`: The domain of the task (e.g., "finance", "medical")
  - `tech_savviness`: User's technical expertise ("LOW", "MEDIUM", "HIGH")
- `scratchpad`: List of previous actions, each containing:
  - `content`: Description of the action taken
- `elements`: List of UI elements, each containing:
  - `x`: X coordinate (integer)
  - `y`: Y coordinate (integer)
  - `width`: Element width (integer)
  - `height`: Element height (integer)
  - `text`: Element text content
  - `type`: Element type (e.g., "button", "textbox", "link")
  - `selector`: Unique identifier for the element
- `prev_action`: Object containing:
  - `action`: Previous action type ("click", "hover", "type")
  - `bounding`: Previous element interacted with (Element object)

##### Example Request:
```json
{
    "task": "Submit the registration form",
    "screenshot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...",
    "previous_screenshot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...",
    "context": {
        "age": 45,
        "domain_familiarity": "intermediate",
        "domain": "finance",
        "tech_savviness": "MEDIUM"
    },
    "scratchpad": [
        {"content": "Found the registration form"},
        {"content": "Filled in personal information"}
    ],
    "elements": [
        {
            "x": 100,
            "y": 200,
            "width": 150,
            "height": 40,
            "text": "Submit",
            "type": "button",
            "selector": "submit-button-1"
        }
    ],
    "prev_action": {
        "action": "hover",
        "bounding": {
            "x": 100,
            "y": 200,
            "width": 150,
            "height": 40,
            "text": "Submit",
            "type": "button",
            "selector": "submit-button-1"
        }
    }
}
```

#### Response Format
```json
{
    "selected_element": {
        "x": integer,
        "y": integer,
        "width": integer,
        "height": integer,
        "text": string,
        "type": string,
        "selector": string
    },
    "action": string,
    "reasoning": string
}
```

##### Field Details:
- `selected_element`: Object containing information about the chosen element
  - `x`: X coordinate
  - `y`: Y coordinate
  - `width`: Element width
  - `height`: Element height
  - `text`: Element text content
  - `type`: Element type
  - `selector`: Unique identifier
- `action`: The action to take ("click", "hover", "type", "backtrack")
- `reasoning`: Detailed explanation of the decision, including:
  - Discoverability score and reasoning
  - Understandability score and reasoning
  - Semantic relevance score and reasoning

##### Example Response:
```json
{
    "selected_element": {
        "x": 100,
        "y": 200,
        "width": 150,
        "height": 40,
        "text": "Submit",
        "type": "button",
        "selector": "submit-button-1"
    },
    "action": "click",
    "reasoning": "Final Score: 0.85 for button\n1. Discoverability (0.90): High contrast, good size...\n2. Understandability (0.95): Generic UI element...\n3. Semantic Relevance (0.75): Directly related to task..."
}
```

#### Error Responses

##### Invalid Base64 Image (400 Bad Request)
```json
{
    "detail": "Invalid base64 image data"
}
```

##### Missing Required Fields (422 Unprocessable Entity)
```json
{
    "detail": [
        {
            "loc": ["body", "task"],
            "msg": "field required",
            "type": "value_error.missing"
        }
    ]
}
```

##### Server Error (500 Internal Server Error)
```json
{
    "detail": "An error occurred: [error details]\nTraceback:\n[stack trace]"
}
```

## Usage Examples

### Basic Request
```python
import requests
import base64

# Read image file
with open("screenshot.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/s2",
    json={
        "task": "Submit the registration form",
        "screenshot": encoded_string,
        "context": {
            "age": 45,
            "domain_familiarity": "intermediate",
            "domain": "finance",
            "tech_savviness": "MEDIUM"
        },
        "elements": [
            {
                "x": 100,
                "y": 200,
                "width": 150,
                "height": 40,
                "text": "Submit",
                "type": "button",
                "selector": "submit-button-1"
            }
        ]
    }
)

print(response.json())
```

## Notes
- The API is stateless - all necessary context should be passed in the request
- The scratchpad field maintains a history of actions and their content
- Screenshots can be provided either as base64 encoded strings or URLs
- The API considers user age, domain familiarity, and tech savviness for personalized decisions
- The reasoning field provides detailed explanation of the decision-making process
- Actions can be "click", "hover", "type", or "backtrack" depending on the context 