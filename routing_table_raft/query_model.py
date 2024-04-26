import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_dir):
    """Load the tokenizer and model from the specified directory."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()  # Set the model to evaluation mode
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Move model to the appropriate device
    return model, tokenizer

def generate_answer(model, tokenizer, question, max_length=128, num_beams=5):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output = model.generate(
        **inputs, 
        max_new_tokens=512, 
        num_beams=num_beams, 
        no_repeat_ngram_size=2,
        temperature=0.5,  # Lower temperature to reduce randomness
        top_k=50,  # Narrow down top-k choices
        top_p=0.95,  # Narrow down top-p cumulative probability
        early_stopping=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
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
        "A default route is represented as 0.0.0.0/0 - could you tell me what my default route is and what the next hop and outgoing interface is please? Do not be verbose."
    ]

    # Run the model on the example questions
    for question in questions:
        answer = generate_answer(model, tokenizer, question)
        print(f"Question: {question}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
