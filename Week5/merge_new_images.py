import json
import sys

def merge_train_data(main_file, new_items_file, output_file="DFDataSplit.json"):
    # Load the main dataset
    with open(main_file, "r") as f:
        data = json.load(f)

    # Load the new train items
    with open(new_items_file, "r") as f:
        new_train_items = json.load(f)

    if "train" not in data:
        print("Error: 'train' key not found in the main JSON file.")
        return

    # Append new items
    data["train"].extend(new_train_items)

    # Save the updated JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f" Successfully updated '{main_file}' and saved as '{output_file}'.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_train_data.py main_data.json new_train_items.json [output_file.json]")
    else:
        main_file = sys.argv[1]
        new_items_file = sys.argv[2]
        merge_train_data(main_file, new_items_file)
