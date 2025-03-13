from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image
import re

from models.ui_ranker import get_next_element

app = FastAPI(
    title="System 2 API",
    description="API for simulating System 2 reasoning in web interaction decisions",
    version="0.1.0"
)


class Element(BaseModel):
    x: int
    y: int
    width: int
    height: int
    text: str
    type: str
    selector: str
    

class System2Request(BaseModel):
    task: str
    previous_screenshot: str | None = None
    screenshot: str
    context: Dict[str, Any]
    scratchpad: List[Dict[str, str]] = []
    elements: List[Element] = []
    prev_action: Dict[str, str | Element] = {}
    
    def get_images(self) -> tuple[Image.Image, Image.Image | None]:
        """Convert both screenshots to PIL Image objects
        Returns:
            tuple: (current_image, previous_image)
            where previous_image may be None if no previous screenshot provided
        """
        def parse_single_image(img_data: str) -> Image.Image:
            # Check if it's a URL
            if img_data.startswith(('http://', 'https://')):
                import requests
                response = requests.get(img_data)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            
            # Check if it's a base64 string
            if img_data.startswith('data:image'):
                # Remove data URL prefix if present
                base64_data = re.sub('^data:image/.+;base64,', '', img_data)
            else:
                base64_data = img_data

            try:
                # Decode base64 string
                image_data = base64.b64decode(base64_data)
                return Image.open(BytesIO(image_data))
            except (base64.binascii.Error, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid base64 image data"
                )

        try:
            # Always parse current screenshot
            current_image = parse_single_image(self.screenshot)
            
            # Parse previous screenshot if provided
            previous_image = None
            if self.previous_screenshot:
                previous_image = parse_single_image(self.previous_screenshot)
            
            return current_image, previous_image
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process screenshots: {str(e)}"
            )


class System2Response(BaseModel):
    selected_element: Element
    action: str
    reasoning: str


@app.post("/s2", response_model=System2Response)
async def system2(request: System2Request):
    try:
        # Get the screenshots as PIL Image objects
        current_image, previous_image = request.get_images()
        
        element, action, reasoning = get_next_element(current_image, request.task, request.context, request.scratchpad, request.elements, request.prev_action)
        
        # For now, return a placeholder response
        return System2Response(
            selected_element=element,
            action=action,
            reasoning=reasoning
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions (like 400s) directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 