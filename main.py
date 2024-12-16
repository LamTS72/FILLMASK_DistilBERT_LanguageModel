import gradio as gr
from torch import topk
from transformers import pipeline


model_path = "Chessmen/mask_lm"
mask_predict = pipeline(
    "fill-mask",
    model=model_path,
)
def predict(text):
    outputs = mask_predict(text, top_k=1)

    return f"{outputs[0]['sequence']}"
        

textbox = gr.Textbox(label="Type your string with [MASK]:", placeholder="[MASK] token", lines=2)

gr.Interface(fn=predict, inputs=textbox, outputs="text").launch()