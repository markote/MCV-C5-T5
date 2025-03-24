import json
import nltk
from collections import Counter
import pickle

# Ensure NLTK punkt tokenizer is downloaded
nltk.download('punkt')

def save_set_as_pickle(my_set, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(my_set, f)

# Function to read the JSON file and process the data
def process_json_file(file_path):
    # Step 1: Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Step 2: Extract titles (first element in each pair)
    titles = []
    for dataset in ["train", "eval", "test"]:
        extra_titles = [title for title, _ in data[dataset]]
        titles = extra_titles + titles
    
    # Step 3: Tokenize each title into words and build the vocabulary
    vocabulary = []
    for title in titles:
        words = nltk.word_tokenize(title)  # Tokenize and make everything lowercase
        vocabulary = vocabulary + words  # Update vocabulary count with each tokenized word

    vocabulary.append(" ")
    vocabulary.append("<EOS>")
    vocabulary.append("<SOS>")
    vocabulary.append("<PAD>")

    return set(vocabulary)

# Main execution
if __name__ == "__main__":
    input_file = "/ghome/c5mcv05/image_captioning_dataset/DataSplit.json"  # Path to the input JSON file
    output_file = "./vocabulary.pkl"  # Path to save the vocabulary

    # Process the JSON file and build the vocabulary
    vocab = process_json_file(input_file)

    # Save the vocabulary to a file
    save_set_as_pickle(vocab, output_file)
    print(f"Size of vocab: {len(vocab)}, vocab: {vocab}")
    print(f"Test: {' ' in vocab}, {'<EOS>' in vocab}, {'<PAD>' in vocab}, {'<SOS>' in vocab}")
    print(f"Vocabulary built and saved to {output_file}")
