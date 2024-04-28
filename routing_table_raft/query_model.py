import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Define the directory where your fine-tuned model is saved
    model_dir = "./phi2-routing-table"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return  # If the model fails to load, exit the function

    # Check if tokenizer and model's vocab size match
    if len(tokenizer) != model.config.vocab_size:
        print("Resizing token embeddings to match tokenizer's vocabulary size.")
        model.resize_token_embeddings(len(tokenizer))

    # Save the tokenizer and model again after resizing token embeddings
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    # Set the model to evaluation mode and move it to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Run the model on example questions
    questions = [
        "What can you tell me about my routing table?",
        "What is my default route?",
        "What is the next hop for my default route?",
        "What interface does my default route use?"
    ]

    for question in questions:
        answer = ask_model(question, model, tokenizer, device)
        print(f"Question: {question}\nAnswer: {answer}\n")

def ask_model(question, model, tokenizer, device, max_length=512, num_beams=5):
    """Generate answers using the fine-tuned model."""
    inputs = tokenizer.encode_plus(
        question, return_tensors="pt", add_special_tokens=True,
        max_length=max_length, padding="max_length", truncation=True,
        return_attention_mask=True
    )

    # Move input tensors to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,  # Set the maximum length for generated tokens
            num_beams=num_beams,  # Set the number of beams for beam search
            early_stopping=True  # Enable early stopping
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    main()
