import os
from datasets import load_dataset, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd

def load_and_prepare_data(csv_path):
    # Load the processed health data CSV
    df = pd.read_csv(csv_path)
    
    # Ensure the CSV has a 'cleaned_text' column (or adjust if necessary)
    if 'cleaned_text' not in df.columns:
        raise ValueError("The CSV must contain a 'cleaned_text' column.")
    
    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df[['cleaned_text']])
    # Rename column to 'text' since GPT-2 training expects 'text'
    dataset = dataset.rename_column("cleaned_text", "text")
    return dataset

def fine_tune_gpt2(train_dataset, val_dataset, output_dir="model/genai_post_model"):
    model_name = "gpt2"  # or "gpt2-medium" if you have enough resources
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Modified tokenization function: convert each text example to str
    def tokenize_function(examples):
        texts = [str(text) for text in examples["text"]]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Use DataCollatorForLanguageModeling to set labels automatically
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,  # Adjust as needed; start small for testing
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    trainer.train()
    # Save the fine-tuned model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete. Model saved to", output_dir)

if __name__ == "__main__":
    # Determine project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Path to the processed CSV file
    csv_path = os.path.join(project_root, "data", "processed", "health_data_processed.csv")
    
    # Load the dataset
    dataset = load_and_prepare_data(csv_path)
    
    # Split dataset into training and validation (e.g., 90/10 split)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    # Fine-tune GPT-2 on the health data
    fine_tune_gpt2(train_dataset, val_dataset)
