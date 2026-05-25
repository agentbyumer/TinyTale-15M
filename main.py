import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def main():
    # Ensure CUDA is active
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: CUDA not found. Training will be extremely slow!")

    # -----------------------------------------------------------------
    # PHASE 1 & 2: DATA ACQUISITION & OFFLINE TOKENIZATION
    # -----------------------------------------------------------------
    TOKENIZED_DATA_DIR = "./tokenized_dataset"

    # To save time on subsequent runs, we check if the tokenized data already exists on disk
    if not os.path.exists(TOKENIZED_DATA_DIR):
        print("\n--- Phase 1: Downloading Raw Dataset to Disk ---")
        raw_dataset = load_dataset("roneneldan/TinyStories", split="train")
        
        # For a thorough but manageable run, let's select the first 200,000 stories
        print("Selecting a 200,000 story subset for training...")
        dataset_subset = raw_dataset.select(range(200000))

        print("\n--- Phase 2: Tokenizing Dataset Offline ---")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=256)

        # num_proc=4 parallelizes tokenization across CPU cores
        print("Processing text into token IDs...")
        tokenized_dataset = dataset_subset.map(
            tokenize_function, 
            batched=True, 
            num_proc=4, 
            remove_columns=["text"]
        )
        
        print(f"Saving pre-processed dataset to: {TOKENIZED_DATA_DIR}")
        tokenized_dataset.save_to_disk(TOKENIZED_DATA_DIR)
    else:
        print(f"\nFound existing tokenized dataset at {TOKENIZED_DATA_DIR}. Loading from disk...")
        tokenized_dataset = load_from_disk(TOKENIZED_DATA_DIR)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------
    # PHASE 3: ARCHITECTURE INITIALIZATION (FROM SCRATCH)
    # -----------------------------------------------------------------
    print("\n--- Phase 3: Initializing Model Architecture from Scratch ---")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=256,
        n_embd=384,
        n_layer=6,
        n_head=6,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # This allocates completely random weights across matrices
    model = GPT2LMHeadModel(config)
    print(f"Model 'TinyTale-15M' initialized successfully.")
    print(f"Total Trainable Parameters: {model.num_parameters() / 1e6:.2f}M")

    # -----------------------------------------------------------------
    # PHASE 4: THE COMPUTE LOOP
    # -----------------------------------------------------------------
    print("\n--- Phase 4: Launching Training Loop on A10 GPU ---")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./tiny_tale_checkpoints",
        per_device_train_batch_size=64,  # High batch size leverages A10 VRAM smoothly
        learning_rate=5e-4,              # Aggressive learning rate for fast convergence
        weight_decay=0.01,
        max_steps=3000,                  # Runs long enough to see true linguistic structure form
        fp16=True,                       # MUST BE TRUE to activate Ampere Tensor Cores
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        report_to="none"                 # Avoids telemetry overhead
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Fire the GPU engines
    trainer.train()

    # -----------------------------------------------------------------
    # PHASE 5: SAVING ARTIFACTS & TESTING INFERENCE
    # -----------------------------------------------------------------
    print("\n--- Phase 5: Saving Final Model Weights ---")
    FINAL_MODEL_DIR = "./TinyTale-15M"
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Weights and configurations saved completely to {FINAL_MODEL_DIR}")

    print("\n--- Generation Test (Inference) ---")
    prompt = "Once upon a time, a small puppy found a shiny key"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device)
    
    output_tokens = model.generate(
        **inputs, 
        max_length=100, 
        do_sample=True, 
        temperature=0.8, 
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    print(f"\n[Prompt]: {prompt}")
    print(f"[Generated Text]:\n{tokenizer.decode(output_tokens[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()