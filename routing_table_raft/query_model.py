import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def load_model(model_dir):
    print("Loading model from:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # If you know the correct config settings, you can create a config object manually
    config = AutoConfig.from_pretrained(model_dir, vocab_size=50296)
    
    # Load the model with the manually created config
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config)
    
    # Check and resize model embeddings if necessary
    if tokenizer.vocab_size != model.config.vocab_size:
        print("Resizing model embeddings to match tokenizer vocab size.")
        model.resize_token_embeddings(tokenizer.vocab_size)
    
    print("Loaded tokenizer vocab size:", tokenizer.vocab_size)
    print("Loaded model vocab size from config:", model.config.vocab_size)
    print("Loaded model embedding size:", model.get_input_embeddings().num_embeddings)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
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
