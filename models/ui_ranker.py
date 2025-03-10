import os
from dotenv import load_dotenv
from models.op_utils.omniparser import Omniparser

load_dotenv()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class UIRanker:
    def get_elements_from_image(self, screenshot_image):
        op_config = {
            'som_model_path': os.path.join(BASE_PATH, os.getenv('SOM_MODEL_PATH')),
            'caption_model_name': os.getenv('CAPTION_MODEL_NAME', 'florence2'),
            'caption_model_path': os.path.join(BASE_PATH, os.getenv('CAPTION_MODEL_PATH')),
            'BOX_TRESHOLD': float(os.getenv('BOX_THRESHOLD', 0.05)),
        }
        omniparser = Omniparser(op_config)
        _, parsed_content_list = omniparser.parse(screenshot_image)
        elements_data = [
            {
                "type": item["type"],
                "text": item["content"],
                "bounds": {
                    "x1": min(item["bbox"][0], item["bbox"][2]),
                    "x2": max(item["bbox"][0], item["bbox"][2]),
                    "y1": min(item["bbox"][1], item["bbox"][3]),
                    "y2": max(item["bbox"][1], item["bbox"][3])
                }
            }
            for item in parsed_content_list
        ]
        return elements_data


    def calculate_discoverability_scores(self, elements, context):
        pass


    def calculate_understandability_scores(self, elements, context):
        pass


    def calculate_semantic_relevance_scores(self, elements, task):
        pass
        

    def rank_elements(self, screenshot_image, task, context, scratchpad):
        elements = self.get_elements_from_image(screenshot_image)
        
        # Discoverability
        discoverability_scores = self.calculate_discoverability_scores(elements, context)
        
        # Understandability
        understandability_scores = self.calculate_understandability_scores(elements, context)
        
        # Semantic relevance
        semantic_relevance_scores = self.calculate_semantic_relevance_scores(elements, task)
        
        # Weighted average of the scores
        scores = []
        for i in range(len(elements)):
            scores.append(
                (
                    0.2 * discoverability_scores[i] +
                    0.4 * understandability_scores[i] +
                    0.4 * semantic_relevance_scores[i]
                )
            )
        
        # makes more sense if we check all elements crossing a threshold, 
        # and then eliminate the ones that are already in the scratchpad(i.e false positives).
        # If all of the candidates after threshold are eliminated, then just tell to go back.
        
        return scores
        

def get_next_element(screenshot_image, task, context, scratchpad):
    ui_ranker = UIRanker()    
    return ui_ranker.rank_elements(screenshot_image, task, context, scratchpad)[-1]