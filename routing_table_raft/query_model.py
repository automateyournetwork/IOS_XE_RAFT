import torch
from peft import PeftModel
from transformers import AutoTokenizer

# Define the directory where your fine-tuned model is saved
model_dir = "./phi2-routing-table"
base_model = "microsoft/phi-2"
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)


ft_model = PeftModel.from_pretrained(model_dir, "phi2-routing-table")

eval_prompt = "What is my default route?"
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.11)[0], skip_special_tokens=True))