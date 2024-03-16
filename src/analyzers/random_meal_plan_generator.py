"""Class for generating random mealplans"""
from models.meal_plan import MealPlan, GroceryList
from models.dataset import Dataset
import random
class RandomMealPlanAnalyzer:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def generate_meal_plan(self, amount: int):
        recipes = random.sample(self.dataset.recipes, amount)
        grocery_list = GroceryList().from_recipes(recipes)
        estimated_cost = grocery_list.estimated_cost
        return MealPlan(recipes, grocery_list, estimated_cost)