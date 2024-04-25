from typing import List
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class Vitamin:
    def __init__(self, name: str, amount: int, unit):
        self.name = name
        self.amount = amount
        self.unit = unit


class Nutrition:
    def __init__(self, calories: int, vitamins: List[Vitamin]):
        self.calories = calories
        self.vitamins = vitamins

class Ingredient:
    def __init__(self, name: str):
        self.name = name
        self.nutrition = None
        self.cost_per_weight = .5

class Recipe:
    def __init__(self, recipe_number=None, title=None, ingredients=None, directions=None, link=None, source=None, NER=None, total_nutrients={"energy": 0, "fat": 0, "protein": 0, "salt": 0, "saturates": 0, "sugars": 0}):
        self.recipe_number = recipe_number
        self.title = title
        self.ingredients = ingredients
        self.directions = directions
        self.link = link
        self.source = source
        self.total_nutrients = total_nutrients
        # NER values contain better information about the ingredients then ingredients list
        #ingredients list contains measure values.
        #TODO find way to better exact measure values
        self.NER = NER

    def to_numerical_representation(self, encoder=OneHotEncoder(sparse=False), scaler=MinMaxScaler()):
        # One-hot encode ingredients
        encoded_ingredients = encoder.transform([[ingredient] for ingredient in self.NER])
        # Normalize calories
        normalized_calories = scaler.transform([[self.calories]])
        # Combine categorical and numerical representations
        numerical_representation = np.concatenate([encoded_ingredients[0], normalized_calories[0]])
        return numerical_representation


class GroceryList:
    def __init__(self, ingredients: List[Ingredient] = None, estimated_cost: int = None):
        self.ingredients = ingredients
        self.estimated_cost = estimated_cost

    def from_recipes(self, recipes: List[Recipe]):
        ingredients = []
        for recipe in recipes:
            for ingredient in recipe.ingredients:
                ingredients.append(ingredient)
        return GroceryList(ingredients=ingredients)
    
    def get_cost(self):
        estimated_cost = 0
        for ingredient in self.ingredients:
            estimated_cost += ingredient.cost_per_weight
        return estimated_cost
    
    def get_ingredients(self):
        names = []
        for ingredient in self.ingredients:
            names.append(ingredient)
        return names

class MealPlan:
    def __init__(self, recipes: List[Recipe], grocery_list: GroceryList, estimated_cost: int):
        self.recipes = recipes
        self.grocery_list = grocery_list
        self.estimated_cost = self.grocery_list.estimated_cost
   
    def get_recipe_names(self):
        names = []
        for recipe in self.recipes:
            names.append(recipe.title)
        return names

