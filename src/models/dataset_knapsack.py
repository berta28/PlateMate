import json
import os
import requests

class Dataset:
    def __init__(self):
        self.recipes = []

    # Load recipes from JSON file
    def create_recipes_from_csv(self, file_location, num_of_entries):
        if not os.path.exists(file_location):
            response = requests.get('https://data.csail.mit.edu/im2recipe/recipes_with_nutritional_info.json')
            if response.status_code == 200 and response.text.strip():
                with open(file_location, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print("Data successfully downloaded and written to file.")
            else:
                print(f"Failed to download the data: Status code {response.status_code}")
                return

        try:
            with open(file_location, 'r', encoding='utf-8') as f:
                data = json.load(f)[:num_of_entries]
        except json.JSONDecodeError as e:
            print("Failed to decode JSON data:", e)
            return

        self.recipes = [self.create_recipe(recipe) for recipe in data]

    # Construct recipe dictionary from raw data
    def create_recipe(self, recipe_data):
        full_ingredients = [ingredient['text'].strip() for ingredient in recipe_data.get('ingredients', [])]
        ingredients = [ingredient.lower() for ingredient in full_ingredients]
        
        return {
            'name': recipe_data.get('title', 'No title provided'),
            'ingredients': ingredients, 
            'full_ingredients': full_ingredients,
            'nutrient_values': recipe_data.get('nutr_values_per100g', {}),
            'url': recipe_data.get('url', '')
        }
