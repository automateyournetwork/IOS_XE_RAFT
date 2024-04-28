from peft import PeftModel
import torch
from transformers import AutoTokenizer

# Define the directory where your fine-tuned model is saved
model_dir = "./phi2-routing-table"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load the fine-tuned model using PeftModel
# Assuming 'Microsoft/phi-2' is your base model and you have fine-tuned on top of it
ft_model = PeftModel.from_pretrained("Microsoft/phi-2", model_dir)

# Ensure the model is in evaluation mode and move it to the appropriate device
ft_model.eval()
ft_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define your prompt for evaluation
eval_prompt = "The following is a note by Eevee the Dog: # Today I "
model_input = tokenizer(eval_prompt, return_tensors="pt").to(ft_model.device)

# Generate text using the fine-tuned model
with torch.no_grad():
    outputs = ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.11)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
