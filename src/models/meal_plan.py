from typing import List



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
    def __init__(self, name, nutrition, cost_per_weight):
        self.name = name
        self.nutrition = nutrition
        self.cost_per_weight = cost_per_weight

class Recipe:
    def __init__(self, ingredients: List[Ingredient], instructions: str, prep_time, cook_time, utensils_needed, nutrition):
        self.ingredients = ingredients
        self.instructions = instructions
        self.prep_time = prep_time
        self.cook_time = cook_time
        self.utensils_needed = utensils_needed
        self.nutrition = nutrition

class GroceryList:
    def __init__(self, ingredients):
        self.ingredients = ingredients
        self.estimated_cost = self.get_cost()
    def get_cost(self):
        estimated_cost = 0
        for ingredient in self.ingredients:
            estimated_cost += ingredient.cost_per_weight
        return estimated_cost

class MealPlan:
    def __init__(self, recipes, grocery_list, estimated_cost):
        self.recipes = recipes
        self.grocery_list = grocery_list
        self.estimated_cost = estimated_cost
