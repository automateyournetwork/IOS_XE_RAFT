import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_dir):
    """Load the tokenizer and model from the specified directory."""
    print("Loading model from:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    print("Loaded tokenizer vocab size:", len(tokenizer))
    print("Loaded model embedding size:", model.get_input_embeddings().num_embeddings)    
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
        temperature=0.5,
        top_k=50,
        top_p=0.95,
        early_stopping=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def main():
    model_dir = "./phi2-routing-table"
    model, tokenizer = load_model(model_dir)

    print("Tokenizer vocab size:", len(tokenizer))
    print("Model embedding size:", model.get_input_embeddings().num_embeddings)

    assert len(tokenizer) == model.get_input_embeddings().num_embeddings, "Mismatch in tokenizer and model embeddings count"

    questions = ["What is my default route?", "What is the next hop for my default route?", "What interface does my default route use?"]
    for question in questions:
        answer = generate_answer(model, tokenizer, question)
        print(f"Question: {question}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()