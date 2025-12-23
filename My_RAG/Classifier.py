import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Classifier:
    def __init__(self, model_path):
        """
        初始化分類器
        param model_path: my_query_classifier/my_domain_classifier
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"fail to load model")

        self.id2label = self._load_label_mapping(model_path)
        #print(f"{len(self.id2label)} classes")

    def _load_label_mapping(self, model_path):
        raw_id2label = {}
        pkl_path = os.path.join(model_path, "label_mapping.pkl")

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    raw_id2label = pickle.load(f)
            except Exception as e:
                print(f"fail to load label_mapping.pkl")

        if not raw_id2label:
            raw_id2label = self.model.config.id2label

        clean_id2label = {}
        for k, v in raw_id2label.items():
            try:
                clean_id2label[int(k)] = v
            except ValueError:
                continue

        return clean_id2label

    def predict(self, text):
        """
        param text: Query Content
        return: query type/domain (String)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax().item()

        if predicted_class_id in self.id2label:
            return self.id2label[predicted_class_id]
        else:
            return f"Unknown_Label_{predicted_class_id}"
