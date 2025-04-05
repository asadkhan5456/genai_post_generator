import os
import pandas as pd
import re

def clean_text(text):
    """
    Clean the input text by:
    - Converting to lowercase
    - Removing extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_health_data(file_path, output_path):
    """
    Load the raw health data CSV, perform cleaning on the text column, 
    and save the processed data to a new CSV file.
    """
    # Load raw data
    df = pd.read_csv(file_path)
    
    # Determine the appropriate text column.
    if 'content' in df.columns:
        text_col = 'content'
    elif 'text' in df.columns:
        text_col = 'text'
    else:
        raise ValueError("The dataset must contain a 'content' or 'text' column with the article content.")
    
    # Clean the text data
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    # Save the processed data
    df.to_csv(output_path, index=False)
    print(f"Processed health data saved to: {output_path}")

if __name__ == '__main__':
    # Determine the project root by moving up three directories from this file's location
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define input and output file paths
    raw_file = os.path.join(project_root, 'data', 'raw', 'health_data.csv')
    processed_file = os.path.join(project_root, 'data', 'processed', 'health_data_processed.csv')
    
    # Ensure the processed folder exists
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    
    # Run the ETL process
    preprocess_health_data(raw_file, processed_file)
