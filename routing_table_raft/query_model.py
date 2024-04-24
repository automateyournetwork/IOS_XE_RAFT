import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the path to the model directory
model_dir = "./phi2-journal-finetune-2024-04-24-12-09"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Ensure the model is using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure the model is in evaluation mode
model.eval()

def ask_model(question, model, tokenizer, device, max_length=512, num_beams=5):
    # Tokenize the input question
    inputs = tokenizer.encode_plus(
        question, 
        return_tensors="pt", 
        add_special_tokens=True, 
        max_length=max_length, 
        padding="max_length", 
        truncation=True,
        return_attention_mask=True
    )

    # Move tensors to the device (GPU or CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set max_new_tokens to limit the length of the generated sequence
    max_new_tokens = 50  # Adjust as needed for the length of the desired output

    # Generate answer using the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
            num_beams=num_beams, 
            early_stopping=True
        )

    # Decode the generated tokens to a string
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Example questions
questions = [
    "What can you tell me about my routing table?",
    "What is my default route?",
    "What is the next hop for my default route?",
    "What interface does my default route use?"
]

# Iterate through each question and print the answers
for question in questions:
    answer = ask_model(question, model, tokenizer, device)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
