import json
import requests
import pandas as pd
import os

# load this json file from this website https://data.csail.mit.edu/im2recipe/recipes_with_nutritional_info.json

def load_dataset(file_location:str)->pd.DataFrame:
    # Download the JSON file
    if not os.path.exists(file_location):
        response = requests.get('https://data.csail.mit.edu/im2recipe/recipes_with_nutritional_info.json')
        # Write the downloaded content to a file
        with open(file_location, 'w') as f:
            f.write(response.text)
    
    # Load the data from the file
    with open(file_location, 'r') as f:
        data = json.load(f)
        f.close()
    dataset = pd.DataFrame(data)
    return dataset

def add_total_weight_to_dataset(dataset: pd.DataFrame)->pd.DataFrame:
    dataset["total_nutr_values"] = [{} for _ in range(len(dataset))]
    # Calculate total nutrient values
    for index, nutrients in dataset["nutr_values_per100g"].items():
        total_nutr_values = sum(dataset["weight_per_ingr"][index])
        for nutrient_name, value in nutrients.items():
            dataset["total_nutr_values"][index][nutrient_name] = int(value * total_nutr_values / 100)
    return dataset

if __name__ == "__main__":
    dataset = load_dataset("data/recipes_with_nutritional_info.json")
    dataset=add_total_weight_to_dataset(dataset)
    dataset.to_csv("recipes_with_nutritional_info.csv", index=False)