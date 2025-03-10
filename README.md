# System 2 Reasoning API

A FastAPI-based service that simulates System 2 reasoning for web interactions. The API processes screenshots and contextual information to make informed decisions about which elements to interact with on a webpage.

## Features

- Screenshot processing (supports both base64 encoded images and URLs)
- Contextual decision making
- Stateless architecture with scratchpad support for reasoning chains
- Confidence scoring for decisions
- Detailed reasoning explanations

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Service

Start the server:
```bash
python main.py
```

The service will be available at `http://localhost:8000`

You can also access the auto-generated Swagger documentation at `http://localhost:8000/docs`

## Quick Start

Here's a simple example of how to use the API:

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

## Testing

Run the test suite:
```bash
pytest
```

## API Documentation

For detailed API documentation, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

## Project Structure

```
.
├── main.py              # FastAPI application and endpoint definitions
├── requirements.txt     # Project dependencies
├── test_main.py        # Test suite
└── API_DOCUMENTATION.md # Detailed API documentation
```

## Dependencies

- Python 3.8+
- FastAPI
- Pillow (PIL)
- pytest (for testing)
- requests
- uvicorn

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Add your license here] 