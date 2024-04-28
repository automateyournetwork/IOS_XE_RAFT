import os
import json
import wandb
import torch
import logging
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
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize wandb
wandb.login()
os.environ["WANDB_PROJECT"] = "phi2-finetune" if "phi2-finetune" else ""

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["HF_HOME"] = os.getenv('HUGGINGFACE_TOKEN')

def load_embedding_model():
    print("Loading Embeddings Model..")
    #return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})
    return OpenAIEmbeddings()

def load_language_model():
    print("Loading llama3 with LoRA adapters..")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",  # Make sure to use the correct model ID
        trust_remote_code=True,
        torch_dtype=torch.float16,
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
        #self.pyatsjob = run_pyats_job()
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
        #llm = Ollama(model="phi")
        llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
        self.qa = ConversationalRetrievalChain.from_llm(llm, self.vectordb.as_retriever(search_kwargs={"k": 5}))

    def collect_data(self, questions):
        empty_chat_history = []  # Assuming the chain can handle an empty list
        return [(q, self.qa.invoke({"question": q, "chat_history": empty_chat_history})) for q in questions]

    def create_jsonl(self, data_pairs, filename='train_dataset.jsonl'):
        print("saving dataset..")
        formatted_data_pairs = [self.formatting_func(pair) for pair in data_pairs]
        with open(filename, 'w') as file:
            for data in formatted_data_pairs:
                file.write(json.dumps(data) + '\n')

    def formatting_func(self, example):
        # Extract question and answer assuming the example is a tuple where
        # example[0] is the question and example[1] is a dictionary with 'answer' as a key
        question = example[0]
        answer = example[1]['answer']

        # Create the formatted dictionary as per the new structure required
        formatted_example = {
            "messages": [
                {"role": "system", "content": "You are a computer networking expert specializing in network routing tables."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        return formatted_example

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
    # chat_instance = ChatWithRoutingTable()
    # questions = [
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",
    #                  "What is the route preference for the default route?",
    #                  "Is the default route actively used?",
    #                  "What is the source protocol for the default route?",
    #                  "What is the source protocol code for the default route?",
    #                  "What is the next hop IP for the route to 10.0.0.0/24?",
    #                  "Which interface is used for the route to 10.0.0.0/24?",
    #                  "Is the route to 10.0.0.0/24 actively used?",
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",                    
    #                  "What is the source protocol for the route to 10.0.0.0/24?",
    #                  "What is the source protocol code for the route to 10.0.0.0/24?",
    #                  "What is the next hop IP for the route to 10.0.0.1/32?",
    #                  "Which interface is used for the route to 10.0.0.1/32?",
    #                  "Is the route to 10.0.0.1/32 actively used?",
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",
    #                  "What is the source protocol for the route to 10.0.0.1/32?",
    #                  "What is the source protocol code for the route to 10.0.0.1/32?",
    #                  "What is the next hop IP for the route to 10.10.20.0/24?",
    #                  "Which interface is used for the route to 10.10.20.0/24?",
    #                  "Is the route to 10.10.20.0/24 actively used?",
    #                  "What is the source protocol for the route to 10.10.20.0/24?",
    #                  "What is the source protocol code for the route to 10.10.20.0/24?",
    #                  "What is the next hop IP for the route to 10.10.20.48/32?",
    #                  "Which interface is used for the route to 10.10.20.48/32?",
    #                  "Is the route to 10.10.20.48/32 actively used?",
    #                  "What is the source protocol for the route to 10.10.20.48/32?",
    #                  "What is the source protocol code for the route to 10.10.20.48/32?",
    #                  "What is the next hop IP for the route to 10.255.255.0/24?",
    #                  "Which interface is used for the route to 10.255.255.0/24?",
    #                  "Is the route to 10.255.255.0/24 actively used?",
    #                  "What is the source protocol for the route to 10.255.255.0/24?",
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",
    #                  "What is the source protocol code for the route to 10.255.255.0/24?",
    #                  "What is the next hop IP for the route to 10.255.255.9/32?",
    #                  "Which interface is used for the route to 10.255.255.9/32?",
    #                  "Is the route to 10.255.255.9/32 actively used?",
    #                  "What is the source protocol for the route to 10.255.255.9/32?",
    #                  "What is the source protocol code for the route to 10.255.255.9/32?",
    #                  "What is the next hop IP for the route to 192.168.1.0/24?",
    #                  "Which interface is used for the route to 192.168.1.0/24?",
    #                  "Is the route to 192.168.1.0/24 actively used?",
    #                  "What is the source protocol for the route to 192.168.1.0/24?",
    #                  "What is the source protocol code for the route to 192.168.1.0/24?",
    #                  "What is the next hop IP for the route to 192.168.1.1/32?",
    #                  "Which interface is used for the route to 192.168.1.1/32?",
    #                  "Is the route to 192.168.1.1/32 actively used?",
    #                  "What is the source protocol for the route to 192.168.1.1/32?",
    #                  "What is the source protocol code for the route to 192.168.1.1/32?",
    #                  "How many routes are there in the default VRF?",
    #                  "How many active routes are there in the IPv4 address family?",
    #                  "How many static routes are there in the IPv4 address family?",
    #                  "How many connected routes are there in the IPv4 address family?",
    #                  "How many local routes are there in the IPv4 address family?",
    #                  "Which routes have a metric of 0?",
    #                  "Which routes use GigabitEthernet1 as the outgoing interface?",
    #                  "Which routes use Loopback0 as the outgoing interface?",
    #                  "Which routes use VirtualPortGroup0 as the outgoing interface?",
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",                    
    #                  "Which routes use Loopback109 as the outgoing interface?",
    #                  "What are the next hop IPs for all active routes?",
    #                  "What are the outgoing interfaces for all static routes?",
    #                  "What are the route preferences for all connected routes?",
    #                  "What are the source protocols for all local routes?",
    #                  "What are the source protocol codes for all routes in the IPv4 address family?"
    #                  "What are the active protocols used in the routing table?",
    #                  "Which routes have a metric set to non-zero?",
    #                  "What is the most common outgoing interface used in the routing table?",
    #                  "What routes have Loopback interfaces as their outgoing interfaces?",
    #                  "Are there any static routes with a next hop IP address specified?",
    #                  "Which routes are configured with the source protocol as 'local'?",
    #                  "Which routes are configured with the source protocol as 'static'?",
    #                  "What are the IP addresses of all routes marked as 'local'?",
    #                  "How many routes are configured with 'GigabitEthernet1' as their outgoing interface?",
    #                  "What is the route preference for routes with the source protocol 'connected'?",
    #                  "Are there any routes with the outgoing interface 'VirtualPortGroup0' that are inactive?",
    #                  "How many routes use a Loopback interface?",
    #                  "What routes use 'Loopback109' as their outgoing interface?",
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",
    #                  "How many routes have the 'connected' source protocol code?",
    #                  "List all routes that use 'VirtualPortGroup0' as their outgoing interface.",
    #                  "Which routes are marked as active and use 'GigabitEthernet1'?",
    #                  "How many routes are there for each source protocol type?",
    #                  "What are the common characteristics of routes with 'Loopback0' as their outgoing interface?",
    #                  "What is the number of routes marked with the source protocol code 'L'?",
    #                  "What interfaces are used by more than one route?",
    #                  "Are there routes with overlapping IP ranges?",
    #                  "What is the total number of routes configured under the default VRF?",
    #                  "How are the routes distributed among different source protocols?",
    #                  "Which static routes also have a next hop IP address specified?",
    #                  "What are the IP addresses for all routes marked as 'connected'?",
    #                  "Can you list the routes with the highest route preference values?",
    #                  "What is the diversity of outgoing interfaces across all routes?",
    #                  "Which routes have the longest subnet masks?",
    #                  "What routes have the shortest subnet masks?",
    #                  "How many routes are directly connected to a network?",
    #                  "What is the IP address range covered by static routes?",
    #                  "How many routes are there with the outgoing interface as 'GigabitEthernet1' and active status?",
    #                  "What is the distribution of next hop IP addresses in the routing table?",
    #                  "Which routes specify an outgoing interface but no next hop IP?",
    #                  "What are the unique source protocol codes used in the routing table?",
    #                  "How many routes are there with each unique source protocol code?",
    #                  "Which routes are not marked as active?",
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",
    #                  "What are the next hop IP addresses for routes marked with source protocol 'connected'?",
    #                  "What routes have a route preference of 1?",
    #                  "What are the next hop IPs for routes with a route preference higher than 1?",
    #                  "What is the range of route preferences used across all routes?",
    #                  "How many routes have a next hop IP address within the '10.10.20.0/24' subnet?",
    #                  "What is the average metric value across all static routes?",
    #                  "Which routes have undefined or null outgoing interfaces?",
    #                  "Are there any inconsistencies in the route preferences across similar routes?",
    #                  "How many routes have a source protocol code that is not 'C', 'S*', or 'L'?",
    #                  "What are the details of the most recently added route?",
    #                  "Are there any deprecated interfaces still listed in the routing table?",
    #                  "How many routes have been configured with redundancy via multiple next hops?",
    #                  "What are the common next hop IPs used for routes with Loopback interfaces?",
    #                  "Which routes can be considered critical based on their active status and source protocol?",
    #                  "How many routes use a metric higher than 1?",
    #                  "What is the route preference for the default IPv4 route?",
    #                  "How many routes have a next-hop IP address assigned?",
    #                  "Are there any routes with no outgoing interfaces specified?",
    #                  "What is the source protocol code for the 192.168.1.1/32 route?",
    #                  "Which routes have a source protocol code 'S*' and what does it signify?",
    #                  "How many routes use the source protocol 'static' and are also active?",
    #                  "What are the characteristics of routes with a route preference of 1?",
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",                    
    #                  "What is the longest subnet mask used in the routing table?",
    #                  "What is the shortest subnet mask used in the routing table?",
    #                  "How many routes use the outgoing interface 'GigabitEthernet1' with a source protocol 'connected'?",
    #                  "What are the IP addresses for routes that are directly connected networks?",
    #                  "Which routes have the same next hop IP address?",
    #                  "What are the total number of interfaces used by all routes in the routing table?",
    #                  "How many routes have a source protocol 'local' and are also active?",
    #                  "What are the different source protocol codes used in the routing table?",
    #                  "Are there routes configured without a next hop IP address?",
    #                  "What is the total number of routes that use a loopback interface as the outgoing interface?",
    #                  "Which routes use an outgoing interface but have no active status?",
    #                  "How many routes are marked with the source protocol 'connected' and have a specified outgoing interface?",
    #                  "What is the average route preference value across all routes?",
    #                  "Which routes use a specific next hop IP within the '10.0.0.0/24' subnet?",
    #                  "How many routes have a metric of zero and are not the default route?",
    #                  "Which routes have overlapping next hop IP addresses?",
    #                  "How many routes are there with the outgoing interface as 'Loopback0' and are inactive?",
    #                  "What is the percentage of active routes in the routing table?",
    #                  "What is the next hop IP for routes that have a metric of zero?",
    #                  "What are the IP addresses of all active routes with the source protocol 'static'?",
    #                  "What are the outgoing interfaces for all routes with a source protocol 'local'?",
    #                  "Which routes have a route preference of 0 and why?",
    #                  "What is the maximum metric used across all routes?",
    #                  "How many routes are configured under VRF named 'default'?",
    #                  "Are there any routes with duplicated configurations?",
    #                  "Which routes have the source protocol 'connected' but no active next hop?",
    #                  "What are the subnet masks for all routes using 'Loopback109' as the outgoing interface?",
    #                  "Which routes are configured with a source protocol 'static' but do not specify a next hop?",
    #                  "How many routes are there with each type of outgoing interface?",
    #                  "Are there routes that specify 'Loopback0' as their outgoing interface but have a next hop IP?",
    #                  "What is the range of metrics used for routes with source protocol 'connected'?",
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",
    #                  "Which routes are configured to be always active?",
    #                  "What are the details of routes that use 'VirtualPortGroup0' as their outgoing interface and are inactive?",
    #                  "How many routes are there without any source protocol code specified?",
    #                  "Are there any routes with a source protocol 'local' that specify a next hop IP?",
    #                  "What are the different types of source protocols used across all routes and how many are there of each type?",
    #                  "Which routes are marked as critical based on their configurations?",
    #                  "How many routes are configured with a source protocol 'connected' and a route preference higher than 1?",
    #                  "Which routes have no active status but have a specified outgoing interface?",
    #                  "What are the most frequently used outgoing interfaces in the routing table?",
    #                  "Which routes have a source protocol code that does not match their actual configurations?",
    #                  "What are the source protocols for routes with a route preference less than the average?",
    #                  "Are there any routes that are deprecated but still listed?",
    #                  "Which routes have undergone the most recent changes?"
    #                  "What are the security levels assigned to routes using 'GigabitEthernet1'?",
    #                  "Which routes are configured with priority over others and why?",
    #                  "Are there any routes linked to specific network policies?",
    #                  "What are the compliance requirements for the routes within the '10.0.0.0/24' subnet?",
    #                  "How many routes are tagged with specific administrative tags?",
    #                  "What routing protocols are implemented across different interfaces?",
    #                  "Which routes are affected by routing policies?",
    #                  "Are there backup routes for critical network paths?",
    #                  "How many routes are designed for load balancing?",
    #                  "What is the redundancy configuration for routes using 'Loopback0'?",
    #                  "How are routing updates managed for the '192.168.1.0/24' subnet?",
    #                  "What disaster recovery protocols are associated with active routes?",
    #                  "Which routes integrate with external routing frameworks?",
    #                  "How many routes have customized traffic engineering tags?",
    #                  "What is the error rate reported on routes using 'VirtualPortGroup0'?",
    #                  "Are there any encrypted routes and what protocols are used?",
    #                  "What is the average packet loss on routes with a source protocol of 'connected'?",
    #                  "Which routes have automation scripts configured and what are their purposes?",
    #                  "How many routes are part of a segregated network segment?",
    #                  "What are the maintenance windows for critical routes?",
    #                  "Which routes have configurable alerts set for performance degradation?",
    #                  "Are there any routes that require manual intervention for changes?",
    #                  "What is the failover time for the default route?",
    #                  "How is traffic prioritized in routes with high data flow?",
    #                  "What monitoring tools are integrated with the routing table?",
    #                  "Are there any routes that are deprecated but still operational?",
    #                  "How many routes are part of the QoS (Quality of Service) framework?",
    #                  "What are the latency benchmarks for routes in the '10.255.255.0/24' subnet?",
    #                  "Which routes have had recent security audits?",
    #                  "How many routes are exposed to public access?",
    #                  "Which routes support multicast traffic?",
    #                  "What are the bandwidth limitations for the '10.10.20.0/24' route?",
    #                  "Are there specific routes designated for VOIP traffic?",
    #                  "What are the backup strategies for routes with a source protocol 'static'?",
    #                  "How are route conflicts resolved in the current network configuration?",
    #                  "Which routes are optimized for cloud connectivity?",
    #                  "What cybersecurity measures are implemented on routes handling sensitive data?",
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",
    #                  "How many routes are configured with dynamic routing protocols?",
    #                  "What are the VLAN configurations for routes using 'VirtualPortGroup0'?",
    #                  "Are there any routes that bypass standard network security checks?",
    #                  "How is network segmentation enforced through routing policies?",
    #                  "What is the impact of a single point of failure in the routing table?",
    #                  "Which routes are prioritized in network congestion scenarios?",
    #                  "How many routes are designed to handle peak traffic loads?",
    #                  "What are the disaster recovery plans for routes marked as 'critical'?",
    #                  "Are there any routes that use legacy protocols?",
    #                  "What is the data integrity protocol for routes transferring confidential information?",
    #                  "How many routes have detailed logging enabled for auditing purposes?",
    #                  "Which routes have network health indicators integrated?",
    #                  "How is network performance measured across different routes?"                    
    #                  "What is the default route in the routing table?",
    #                  "What is the next hop IP for the default route?",
    #                  "Which interface is used for the default route?",
    #                  "What metric is used for the default route?",                    
    #              ]
    # data_pairs = chat_instance.collect_data(questions)
    # chat_instance.create_jsonl(data_pairs)
    # Initialize model and tokenizer
    model = load_language_model()
    base_model_name = "llama3"
    run_name = f"{base_model_name}-routing-table"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    print("Tokenizer vocab size before:", len(tokenizer))
    # Add a pad token if it does not exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        print("Adjusted Model embedding size:", model.get_input_embeddings().num_embeddings)

    print("Tokenizer vocab size after:", len(tokenizer))

    # Confirm that tokenizer and model's vocab size match before ending the training script
    assert len(tokenizer) == model.config.vocab_size, "Mismatch in tokenizer and model embeddings count after training"
    print("Training and saving completed successfully.")

    # Display the tokenizer and model sizes to confirm correct setup
    print("Tokenizer vocab size:", len(tokenizer))
    print("Model embedding size:", model.get_input_embeddings().num_embeddings)

    # Load and prepare dataset
    train_dataset = load_dataset('json', data_files='train_dataset.jsonl', split='train')
    print(train_dataset)  # Check initial dataset structure

    def tokenize_and_format(examples):
        # This assumes 'examples' is a batch from 'train_dataset'
        inputs = []
        outputs = []

        # Iterate through each example in the provided batch
        for example in examples['messages']:
            # Filter and concatenate messages based on their roles
            input_text = " ".join(msg['content'] for msg in example if msg['role'] != 'assistant')
            output_text = " ".join(msg['content'] for msg in example if msg['role'] == 'assistant')
            inputs.append(input_text)
            outputs.append(output_text)

        # Tokenize inputs and outputs
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        model_outputs = tokenizer(outputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

        # Return a properly structured dictionary
        return {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "labels": model_outputs.input_ids  # Labels are typically aligned with outputs
        }

    # Usage of the function in dataset mapping should include error handling
    try:
        tokenized_train_dataset = train_dataset.map(
            tokenize_and_format,
            batched=True
        )
    except Exception as e:
        print(f"An error occurred during tokenization/formatting: {str(e)}")

    # Setup Accelerator for distributed training
    accelerator = Accelerator()
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Configure training arguments
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
        evaluation_strategy="no",
        eval_steps=25,
        do_eval=False,
        report_to="wandb",
        run_name=run_name
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Start training
    assert len(tokenizer) == model.get_input_embeddings().num_embeddings, "Mismatch in tokenizer and model embeddings count"
    
    print("Training model...")
    trainer.train()

    print("Saving tokenizer and model to:", f"./{run_name}")
    tokenizer.save_pretrained(f"./{run_name}")
    model.save_pretrained(f"./{run_name}")
    