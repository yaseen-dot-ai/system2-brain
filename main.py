import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
import asyncio
from PIL import Image
import re
import traceback

from models.ui_ranker import UIRanker
from models.utils import chat_anthropic

app = FastAPI(
    title="System 2 API",
    description="API for simulating System 2 reasoning in web interaction decisions",
    version="0.1.0"
)


ui_ranker = UIRanker()
async def get_next_element(screenshot_image, previous_screenshot_image, task, context, scratchpad, elements, prev_action, ui_ranker=ui_ranker):
    if scratchpad:
        scratchpad_str = '\n------------\n'.join([f'{action["content"]}' for i, action in enumerate(scratchpad)])
        prompt = f"""You are a {context["age"]} years old {context["domain_familiarity"]} level {context["domain"]}, with {context["tech_savviness"]} tech savviness, navigating a UI to complete a specific task. 

TASK OBJECTIVE:
```
{task}
```

INTERACTION HISTORY:
This is your report of the actions you have performed:
```
{scratchpad_str}
```

CURRENT SITUATION:
Looking at the current screenshot and considering the interaction history:
1. Does the current page still lead toward the task objective?
2. Have we moved away from the logical path to complete the task?
3. Would a typical {context["domain_familiarity"]} {context["domain"]} with {context["tech_savviness"]} tech savviness realize they need to backtrack?

DECISION NEEDED:
Should we continue from this point or backtrack to a previous state?"""
        
        class BacktrackDecision(BaseModel):
            '''Evaluation of whether to continue or backtrack based on task progress'''
            backtrack: bool = Field(
                description="True if you should return to a previous state (wrong path/dead end), False if current path still leads to objective"
            )
            confidence: float = Field(
                description="Confidence in this decision (0.0-1.0). Consider your expertise and clarity of the situation",
                ge=0.0,
                le=1.0
            )
            reasoning: str = Field(
                description="Brief explanation of why this decision makes sense for your current situation"
            )
            
        response = await chat_anthropic(prompt, screenshot_image, output_schema=BacktrackDecision)
    
        if response.backtrack and response.confidence > 0.5:
            return None, "backtrack", response.reasoning
    
    last_action = prev_action.get("action", None)
    last_element = prev_action.get("bounding", None)
    
    ranked_elements = await ui_ranker.rank_elements(screenshot_image, previous_screenshot_image, elements, task, context, last_element)
    output_element = ranked_elements[0]
    
    if last_action == "hover":
        if last_element["element_id"] == output_element["element_id"]:
            action = "click"
        else:
            action = "hover"
    elif last_action == "click":
        if last_element["element_id"] == output_element["element_id"] and output_element["type"] == "textbox":
            action = "type"
        else:
            action = "click"
    else:
        action = "click"
        
    
    reasoning = output_element["final_reasoning"]
    print("reasoning: ", reasoning)
    
    print("output of ui_ranker.rank_elements:\n", json.dumps(output_element, indent=4))
    return output_element, action, reasoning


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
        
        element, action, reasoning = await get_next_element(current_image, previous_image, request.task, request.context, request.scratchpad, request.elements, request.prev_action)
        
        # For now, return a placeholder response
        return System2Response(
            selected_element=element,
            action=action,
            reasoning=reasoning
        )
        
    except HTTPException as http_exc:
        raise  # Re-raise HTTP exceptions (like 400s) directly
    except Exception as e:
        # Capture the traceback
        tb = traceback.format_exc()
        # Log or print the traceback if needed
        print(tb)  # You can also use logging instead of print
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}\nTraceback:\n{tb}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 