import numpy as np
import json
import random


def get_titles(list_title_image):
    return set([title_image_path[0] for title_image_path in list_title_image])

def get_titles_dict(dict_data):
    set1 = get_titles(dict_data["train"])
    set2 = get_titles(dict_data["eval"])
    set3 = get_titles(dict_data["test"])

    joined = set1.union(set2)
    joined = joined.union(set3)
    return joined

def prompt_generator(title, number_random_prompts=2):
    numbers = random.sample(range(4), number_random_prompts)
    conditions = [
        ", surounded by different food",
        ", surounded by the same food",
        ", seen from far away",
        ", seen from really close",
        ]
    prompt = "A dish or a drink named ###FOOD######condition###."
    prompts = []
    prompts.append([title, prompt.replace("###FOOD###", title).replace("###condition###","")])

    for i in numbers:
        prompts.append([title, prompt.replace("###FOOD###", title).replace("###condition###", conditions[i])])
    
    return prompts
    


if __name__ == "__main__":
    with open('./FilteredDataSplit.json', 'r') as f:
        filtered_data = json.load(f)
        filtered_data = get_titles_dict(filtered_data)
    with open('./DataSplit.json', 'r') as f:
        data = json.load(f)
        data = get_titles_dict(data)

    set_discarded_titles = data - filtered_data
    discarded_titles = list(set_discarded_titles)
    print("discarded titles:", len(discarded_titles))
    
    resulting_prompts = []
    for t in discarded_titles:
        resulting_prompts += prompt_generator(t)
    
    print("diffusion prompts:", len(resulting_prompts))
    with open("DiffusionPrompts.json", "w") as f:
        json.dump(resulting_prompts, f, indent=4) 