from ultralytics.models.sam import SAM3SemanticPredictor
import os
import sys

def main():
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=os.getenv("SAM3_MODEL_PATH", "sam3.pt"),
        half=True,
        save=True,
    )

    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.getenv("SAM3_IMAGE_PATH")

    model_path = overrides["model"]
    if not model_path or not os.path.exists(model_path):
        print(f"提示：未找到模型檔：{model_path}，請設定SAM3_MODEL_PATH或放置sam3.pt於此目錄。")
        return
    if not image_path or not os.path.exists(image_path):
        print("提示：請提供有效影像路徑，例如：python 00_test_field/sam3.py C:\\path\\to\\image.jpg")
        print("或設定環境變數 SAM3_IMAGE_PATH。")
        return

    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(image_path)

    results = predictor(text=["person", "bus", "glasses"])
    results = predictor(text=["person with red cloth", "person with blue cloth"])
    results = predictor(text=["a person"])

if __name__ == "__main__":
    main()
