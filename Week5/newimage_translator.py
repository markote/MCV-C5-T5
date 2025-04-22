import os
import json

# Set your directory path here
directory = "/path/to/your/images"

result = []

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        # Split into number and name
        try:
            number, name_with_ext = filename.split("_", 1)
            name = name_with_ext.rsplit(".", 1)[0]  # Remove .jpg
            result.append([name, filename])
        except ValueError:
            # Skip files that don't match the expected pattern
            continue

# Write to a JSON file (optional)
with open("image_data.json", "w") as f:
    json.dump(result, f, indent=4)

# Print result
print(json.dumps(result, indent=4))
