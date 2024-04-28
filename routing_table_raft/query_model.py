import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_dir):
    """Load the tokenizer and model from the specified directory and resize embeddings if necessary."""
    print("Loading model from:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # Immediately check if token embeddings need resizing
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        print("Resizing model embeddings to match tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
    
    print("Loaded tokenizer vocab size:", len(tokenizer))
    print("Loaded model embedding size:", model.get_input_embeddings().num_embeddings)    
    model.eval()  # Set the model to evaluation mode
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Move model to the appropriate device
    return model, tokenizer

def main():
    model_dir = "./phi2-routing-table"
    model, tokenizer = load_model(model_dir)

    # Verify embedding sizes match
    assert len(tokenizer) == model.get_input_embeddings().num_embeddings, "Mismatch in tokenizer and model embeddings count"

    questions = ["What is my default route?", "What is the next hop for my default route?", "What interface does my default route use?"]
    for question in questions:
        answer = generate_answer(model, tokenizer, question)
        print(f"Question: {question}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
