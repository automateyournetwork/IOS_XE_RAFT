import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_dir):
    """Load the tokenizer and model from the specified directory."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()  # Set the model to evaluation mode
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Move model to the appropriate device
    return model, tokenizer

def generate_answer(model, tokenizer, question, max_length=512, num_beams=5):
    """Generate an answer from the model based on the input question."""
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}  # Ensure inputs are on the correct device
    output = model.generate(
        **inputs, 
        max_length=max_length, 
        num_beams=num_beams, 
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)  # Decode the output tokens to string
    return answer

def main():
    # Define the directory where your fine-tuned model is saved
    model_dir = "./phi2-routing-table"
    model, tokenizer = load_model(model_dir)

    # Example questions to test the model
    questions = [
        "What can you tell me about my routing table?",
        "What is my default route?",
        "What is the next hop for my default route?",
        "What interface does my default route use?"
    ]

    # Run the model on the example questions
    for question in questions:
        answer = generate_answer(model, tokenizer, question)
        print(f"Question: {question}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
