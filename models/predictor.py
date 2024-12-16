from preprocessing import Preprocessing
from transformers import (
    pipeline,
    AutoModelForMaskedLM
)
import torch

class Predictor():
    def __init__(self):
        self.process = Preprocessing(flag_traning=False)
        self.model = self.load_model()

    def load_model(self):
        model = AutoModelForMaskedLM.from_pretrained(
            "/kaggle/working/mask_lm",
            use_safetensors=True,
        )
        return model

    def predict(self, sample):
        self.model.eval()
        input = self.process.tokenizer(
            [sample],
            return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**input)
        token_logits = output.logits
        mask_index_token = torch.where(input["input_ids"] == self.process.tokenizer.mask_token_id)[1]
        mask_logit_token = token_logits[0, mask_index_token, :]
        top5_tokens = torch.topk(mask_logit_token,
                                 5,
                                 dim=1).indices[0].tolist()
        
        for token in top5_tokens:
            print(f"'>>> {sample.replace(self.process.tokenizer.mask_token, self.process.tokenizer.decode([token]))}'")
            
            
            
if __name__ == "main":
    predict = Predictor()
    predict.predict("This is a great [MASK].")