from pathlib import Path
import torch

MODEL_NAME = "skt/kobert-base-v1"

class KoBERTClassifier:
    """
    fine-tuned model directory가 있으면 이를 사용한다.
    예: models/kobert_category/
    """
    def __init__(self, model_dir="models/kobert_category"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_path = Path(model_dir)
        source = str(model_path) if model_path.exists() else MODEL_NAME

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.eval()

        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        idx = int(probs.argmax())
        return {
            "label": self.id2label.get(idx, str(idx)),
            "confidence": float(probs[idx])
        }
