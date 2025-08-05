import pandas as pd
import os

def add_ids_to_news_csv(input_filepath, output_filepath):
    """
    Reads a news CSV, uses the row index as a unique 'news_id' for each article, 
    and saves the result to a new CSV file.
    """
    print(f"Reading news data from: {input_filepath}")
    
    # Check if the input file exists
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at '{input_filepath}'")
        return

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_filepath)
    
    print(f"Found {len(df)} articles. Generating 'news_id' from index...")
    
    # 1. Reset the index to convert it to a column named 'index'
    df.reset_index(inplace=True)
    
    # 2. Rename the new 'index' column to 'news_id'
    df.rename(columns={'index': 'news_id'}, inplace=True)
    
    # Reorder columns to have 'news_id' first for convenience
    # This assumes 'title', 'full_text', and 'year' exist in the original CSV.
    df = df[['news_id', 'title', 'full_text', 'year']]
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_filepath, index=False)
    
    print(f"Successfully saved news with IDs to: {output_filepath}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Define the path to your original news file
    original_news_file = '../agentsociety/data/news_data/combined_news.csv'
    
    # Define the path for the new file that will be created
    news_file_with_ids = '../agentsociety/data/news_data/news_with_ids.csv'
    
    # Run the preprocessing function
    add_ids_to_news_csv(original_news_file, news_file_with_ids)