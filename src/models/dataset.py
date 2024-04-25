"""This file is used to create classes for the dataset to use with the analyzers"""

from models.meal_plan import MealPlan, Recipe, Ingredient, Nutrition, Vitamin
import pandas as pd
from typing import List
import ast
import copy
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os
import json
import requests

class Dataset:
    def __init__(self, recipes: List[Recipe] = None, ingredients: List[Ingredient]=None, meal_plans: List[MealPlan]=None):
        
        self.recipes = recipes
        self.ingredients = ingredients
        self.meal_plans = meal_plans
        self.ingredient_names = []
    
    def get_recipe_by_title(self, title: str) -> Recipe:
        for recipe in self.recipes:
            if recipe.title == title:
                return recipe
        return None

    def get_ingredient_by_name(self, name: str) -> Ingredient:
        for ingredient in self.ingredients:
            if ingredient.name == name:
                return ingredient
        return None

    def get_meal_plan_by_title(self, title: str) -> MealPlan:
        for meal_plan in self.meal_plans:
            if meal_plan.title == title:
                return meal_plan
        return None
    
    def create_recipes_from_csv(self, file_location: str="data/recipes/full_dataset.csv"):
        recipes = []
        if not os.path.exists(file_location):
            response = requests.get('https://data.csail.mit.edu/im2recipe/recipes_with_nutritional_info.json')
            # Write the downloaded content to a file
            with open(file_location, 'w') as f:
                f.write(response.text)
        
        # Load the data from the file
        with open(file_location, 'r') as f:
            data = json.load(f)
            f.close()
        dataset = pd.DataFrame(data[:3])
    
        dataset["total_nutr_values"] = [{} for _ in range(len(dataset))]
        # Calculate total nutrient values
        for index, nutrients in dataset["nutr_values_per100g"].items():
            total_nutr_values = sum(dataset["weight_per_ingr"][index])
            for nutrient_name, value in nutrients.items():
                dataset["total_nutr_values"][index][nutrient_name] = int(value * total_nutr_values / 100)
        self.dataset = dataset

        for index, row in dataset.iterrows():
            #TODO links to ingredients
            recipe = Recipe(index, row.title, row.ingredients, row.instructions, row.url, '____', [ing['text'] for ing in row.ingredients], row.total_nutr_values)
            recipes.append(recipe)
            #print(recipe.total_nutrients['saturates'])

        self.recipes = recipes

        #create ingredient names list
        self.ingredient_names = self.get_ingredient_names()
        pd.DataFrame(recipes).to_csv("recipes_with_nutritional_info.csv", index=False)

    def filter_by_allergies(self, allergies: List[str]):
        filtered_recipes = copy.copy(self.recipes)
        for recipe in self.recipes:
            for ingredient in recipe.NER:
                if ingredient in allergies:
                    filtered_recipes.remove(recipe)
                    break
        return filtered_recipes
    
    def get_ingredient_names(self):
        ingredient_names = []
        for recipe in self.recipes:
            for ingredient in recipe.NER:
                if ingredient not in ingredient_names:
                    ingredient_names.append(ingredient)
        return ingredient_names

    def get_input_shape(self):
        max_sequence_length = max(len(recipe.NER) for recipe in self.recipes)
        return (max_sequence_length, len(self.ingredient_names))

    def get_num_categories(self):
        return len(self.ingredient_names)
    
    def preprocess_data(self,encoder = OneHotEncoder(sparse=False),scaler = MinMaxScaler()):
        encoded_ingredients = encoder.fit_transform([[ingredient] for recipe in self.recipes for ingredient in recipe.NER])
        # Convert preprocessed data to numpy array
        return np.array(encoded_ingredients)


class IngredientsDataset:
    def __init__(self, ingredients: List[Ingredient] = None):
        self.ingredients = ingredients

    def get_ingredient_by_name(self, name: str) -> Ingredient:
        for ingredient in self.ingredients:
            if ingredient.name == name:
                return ingredient
        return None

    def create_ingredients_from_csv(self, file_path: str="data/recipes/full_dataset.csv"):
        ingredients = []
        df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
        for index, row in df.iterrows():
            name = row['name']
            nutrition = Nutrition(row['calories'], [Vitamin(row['vitamin'], row['amount'], row['unit'])])
            cost_per_weight = row['cost_per_weight']
            ingredient = Ingredient(name, nutrition, cost_per_weight)
            ingredients.append(ingredient)
        self.ingredients = ingredients
        
