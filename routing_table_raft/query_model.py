import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(checkpoint_dir):
    print("Loading model from:", checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model.eval()  # Set the model to evaluation mode
    model.to("cuda")  # Move model to the GPU if available
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
    # Instead of "phi2-routing-table", use the specific checkpoint directory
    checkpoint_dir = "./phi2-routing-table/checkpoint-500"  # Adjust based on the correct path to your checkpoint
    model, tokenizer = load_model(checkpoint_dir)

    # Confirm the model's embeddings match the tokenizer's vocabulary size
    assert len(tokenizer) == model.get_input_embeddings().num_embeddings, "Mismatch in tokenizer and model embeddings count"

    # Ask questions to test the model
    questions = ["What is my default route?", "What is the next hop for my default route?", "What interface does my default route use?"]
    for question in questions:
        answer = generate_answer(model, tokenizer, question)
        print(f"Question: {question}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
