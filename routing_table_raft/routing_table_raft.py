import os
import json
import wandb
import torch
from datetime import datetime
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig

# Initialize wandb
wandb.login()
os.environ["WANDB_PROJECT"] = "phi2-finetune" if "phi2-finetune" else ""

quantization_config = BitsAndBytesConfig()

def load_embedding_model():
    print("Loading Embeddings Model..")
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})

def load_language_model():
    print("Loading Phi-2 with LoRA adapters..")
    model = AutoModelForCausalLM.from_pretrained(
        "phi-2/",  # Make sure to use the correct model ID
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_bits=quantization_config,
        low_cpu_mem_usage=True
    )
    model = attach_lora_adapters(model)
    print_trainable_parameters(model)
    return model

def run_pyats_job():
    print("Running pyATS Jobs")
    os.system("pyats run job show_ip_route_langchain_job.py")

class ChatWithRoutingTable:
    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.pyatsjob = run_pyats_job()
        self.load_text()
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_retrieval_chain()

    def load_text(self):
        print("Loading Text..")
        self.loader = JSONLoader(file_path='Show_IP_Route.json', jq_schema=".info[]", text_content=False)
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        print("Chunking Text..")
        self.text_splitter = SemanticChunker(self.embedding_model)
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        print("Storing in Chroma..")
        self.vectordb = Chroma.from_documents(self.docs, embedding=self.embedding_model)
        self.vectordb.persist()

    def setup_conversation_retrieval_chain(self):
        print("Setup conversation..")
        llm = Ollama(model="phi")
        self.qa = ConversationalRetrievalChain.from_llm(llm, self.vectordb.as_retriever(search_kwargs={"k": 5}))

    def get_default_route_with_rag(self):
        print("Asking What is my default route..")
        return self.qa.invoke("What is my default route?")

    def collect_data(self, questions):
        empty_chat_history = []  # Assuming the chain can handle an empty list
        return [(q, self.qa.invoke({"question": q, "chat_history": empty_chat_history})) for q in questions]

    def create_jsonl(self, data_pairs, filename='train_dataset.jsonl'):
        print("saving dataset..")
        with open(filename, 'w') as file:
            for question, answer in data_pairs:
                file.write(json.dumps({"input": question, "output": answer}) + '\n')

def formatting_func(example):
    return f"### Question: {example['input']}\n### Answer: {example['output']}"

def tokenize_and_format(batch, tokenizer):
    # Ensure all inputs and outputs are strings
    inputs = [str(i) for i in batch['input']]
    outputs = [str(o) for o in batch['output']]

    # Tokenizing inputs and outputs
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    model_outputs = tokenizer(outputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # Return a dictionary matching the expected structure
    return {
        "input_ids": model_inputs['input_ids'],
        "attention_mask": model_inputs['attention_mask'],
        "labels": model_outputs['input_ids']
    }

def attach_lora_adapters(model):
    # Configuring LoRA
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "Wqkv",
            "fc1",
            "fc2",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )
    # Applying LoRA to the model
    model = get_peft_model(model, config)
    return model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# FSDP and Accelerator configuration
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

if __name__ == "__main__":
    chat_instance = ChatWithRoutingTable()
    questions = ["What is my default route?"]
    data_pairs = chat_instance.collect_data(questions)
    chat_instance.create_jsonl(data_pairs)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = load_language_model()

    # Resize model embeddings to account for potential new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Load and prepare dataset
    train_dataset = load_dataset('json', data_files='train_dataset.jsonl', split='train')
    print(train_dataset)  # Check dataset structure
    tokenized_train_dataset = train_dataset.map(
        lambda batch: tokenize_and_format(batch, tokenizer),
        batched=True,
        num_proc=1  # Adjust based on your available CPU cores
    )

    # Setup Accelerator
    accelerator = Accelerator()
    model = accelerator.prepare(model)

    # Configure and start training
    project = "journal-finetune"
    base_model_name = "phi2"
    run_name = f"{base_model_name}-{project}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    training_args = TrainingArguments(
        output_dir=f"./{run_name}",
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=500,
        learning_rate=2.5e-5,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        do_eval=True,
        report_to="wandb",
        run_name=run_name
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Training model...")
    trainer.train()
