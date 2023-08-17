import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import torch
import math


def fine_tune_gpt2(file_path, output_dir="./gpt2-soccer", model_name="gpt2-medium", num_train_epochs=1):
    
    torch.cuda.empty_cache()

    # Split the data into train and validation
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        random.shuffle(lines)
        
        train_size = int(0.9 * len(lines))
        train_lines = lines[:train_size]
        valid_lines = lines[train_size:]
        
        train_file_path = "train_data.txt"
        valid_file_path = "valid_data.txt"
        
        with open(train_file_path, 'w', encoding="utf-8") as train_file:
            train_file.writelines(train_lines)
            
        with open(valid_file_path, 'w', encoding="utf-8") as valid_file:
            valid_file.writelines(valid_lines)

    # Load the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Prepare datasets
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file_path,
        block_size=64
    )
    valid_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=valid_file_path,
        block_size=64
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_steps=100,  
        save_steps=100,
        warmup_steps=10,
        logging_dir='./logs'

    )
    
    # Initialize the Trainer and train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)

    # Clean up temporary train and validation files
    import os
    os.remove(train_file_path)
    os.remove(valid_file_path)

    print(f"Model has been fine-tuned and saved to {output_dir}")

# Example usage
fine_tune_gpt2("/home/mehdi/Twitter-Thesis/GPT-FineTuned-Tweets/input.txt")
