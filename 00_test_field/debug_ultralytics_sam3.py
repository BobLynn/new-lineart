import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
import torch

def test_interactive():
    # Mock image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    try:
        predictor = SAM3SemanticPredictor(overrides={'model': 'checkpoints/sam3.pt'})
        print("Predictor instantiated")
    except Exception as e:
        print(f"Instantiation failed: {e}")
        return

    try:
        predictor.set_image(img)
        print("Image set")
    except Exception as e:
        print(f"set_image failed: {e}")
        
    try:
        # prompt_inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False)
        points = [[50, 50]]
        labels = [1]
        print("Calling prompt_inference...")
        res = predictor.prompt_inference(im=img, points=points, labels=labels)
        print(f"prompt_inference result type: {type(res)}")
        print(f"prompt_inference result: {res}")
        
        if isinstance(res, list):
             print(f"Result is list, length: {len(res)}")
             if len(res) > 0:
                 print(f"First element: {res[0]}")
                 if hasattr(res[0], 'masks'):
                     print(f"Masks: {res[0].masks}")
        else:
             print(f"Result is not list")
             
    except Exception as e:
        print(f"prompt_inference failed: {e}")

if __name__ == "__main__":
    test_interactive()
