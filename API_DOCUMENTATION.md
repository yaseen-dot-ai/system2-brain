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
    "task": string,          // Required: The task that needs to be completed
    "screenshot": string,    // Required: Base64 encoded image or image URL
    "context": object,       // Required: Additional context about the page
    "scratchpad": string    // Optional: Previous thoughts or reasoning steps
}
```

##### Field Details:
- `task`: Description of what needs to be accomplished
- `screenshot`: Can be either:
  - Base64 encoded image (with or without data URL prefix)
  - Direct URL to an image
- `context`: JSON object containing relevant page information
- `scratchpad`: Previous reasoning steps or thoughts (useful for maintaining state)

##### Example Request:
```json
{
    "task": "Click the submit button",
    "screenshot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...",
    "context": {
        "current_url": "https://example.com",
        "page_title": "Test Page"
    },
    "scratchpad": "Previous thoughts: Looking for a blue button..."
}
```

#### Response Format
```json
{
    "selected_element": {
        "id": string,
        "type": string,
        // Additional element properties
    },
    "reasoning": string,
    "confidence": float
}
```

##### Field Details:
- `selected_element`: Object containing information about the chosen element
- `reasoning`: Explanation of why this element was selected
- `confidence`: Float between 0 and 1 indicating confidence in the selection

##### Example Response:
```json
{
    "selected_element": {
        "id": "submit-button",
        "type": "button"
    },
    "reasoning": "Found the primary submit button at the bottom of the form",
    "confidence": 0.95
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
    "detail": "Failed to process screenshot: [error details]"
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
        "task": "Click the submit button",
        "screenshot": encoded_string,
        "context": {
            "current_url": "https://example.com",
            "page_title": "Test Page"
        }
    }
)

print(response.json())
```

### Request with Scratchpad
```python
response = requests.post(
    "http://localhost:8000/s2",
    json={
        "task": "Click the submit button",
        "screenshot": encoded_string,
        "context": {
            "current_url": "https://example.com",
            "page_title": "Test Page"
        },
        "scratchpad": "Previous thoughts: Looking for a blue button..."
    }
)
```

## Notes
- The API is stateless - all necessary context should be passed in the request
- The scratchpad field can be used to maintain reasoning state between requests
- Screenshots can be provided either as base64 encoded strings or URLs
- All responses include confidence scores to help evaluate the reliability of the decision 