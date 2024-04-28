from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and the model
model_dir = "./phi2-routing-table"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Ensure the tokenizer's vocab size matches the model's token embeddings
if len(tokenizer) != model.config.vocab_size:
    print("The tokenizer and model vocab sizes do not match. Resizing model embeddings to match tokenizer.")
    model.resize_token_embeddings(len(tokenizer))

    # Save both the tokenizer and the model again to be sure
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
else:
    print("The tokenizer and model vocab sizes match. No resizing is needed.")
