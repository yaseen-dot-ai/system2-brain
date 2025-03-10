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


class System2Request(BaseModel):
    """Base model for System 2 requests"""
    scratchpad: Optional[str] = Field(None, description="The scratchpad for the user to write down their thoughts")
    task: str  # the task that the user is trying to complete
    screenshot: str = Field(..., description="Base64 encoded screenshot or image URL")
    context: Dict[str, Any]

    def get_image(self) -> Image.Image:
        """Convert the screenshot to a PIL Image object"""
        try:
            # Check if it's a URL
            if self.screenshot.startswith(('http://', 'https://')):
                import requests
                response = requests.get(self.screenshot)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            
            # Check if it's a base64 string
            if self.screenshot.startswith('data:image'):
                # Remove data URL prefix if present
                base64_data = re.sub('^data:image/.+;base64,', '', self.screenshot)
            else:
                base64_data = self.screenshot

            try:
                # Decode base64 string
                image_data = base64.b64decode(base64_data)
                return Image.open(BytesIO(image_data))
            except (base64.binascii.Error, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid base64 image data"
                )
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process screenshot: {str(e)}"
            )


class System2Response(BaseModel):
    """Base model for System 2 responses"""
    selected_element: Dict[str, Any]
    reasoning: str
    confidence: float


@app.post("/s2", response_model=System2Response)
async def system2(request: System2Request):
    """
    System 2 endpoint for determining web interactions.
    
    Args:
        request: System2Request containing screenshot and context
        
    Returns:
        System2Response containing the selected element, reasoning, and confidence
    """
    try:
        # Get the screenshot as a PIL Image
        screenshot_image = request.get_image()
        
        element_id, element_type, confidence, reasoning = get_next_element(screenshot_image, request.task, request.context, request.scratchpad)
        
        # For now, return a placeholder response
        return System2Response(
            selected_element={"id": element_id, "type": element_type},
            reasoning=reasoning,
            confidence=confidence
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions (like 400s) directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 