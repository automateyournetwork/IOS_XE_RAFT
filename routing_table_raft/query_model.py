import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Load model and tokenizer
    model_dir = "./phi2-routing-table"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

    # Ensure tokenizer and model's vocab size match
    if tokenizer.vocab_size != model.config.vocab_size:
        print(f"Resizing model's token embeddings to match tokenizer's vocab size of {tokenizer.vocab_size}.")
        model.resize_token_embeddings(tokenizer.vocab_size)

    # After ensuring the sizes match, save the configurations
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Set model to eval mode and to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Run inference
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
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs.to(device)
    
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    main()
