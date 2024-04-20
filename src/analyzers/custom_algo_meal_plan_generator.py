from models.meal_plan import MealPlan, GroceryList
from models.dataset import Dataset
from user_input.user_input import user_input
import random


class customAlgoMealPlanGenerator:
    def __init__(self, dataset : Dataset, user : user_input):
        self.dataset = dataset
        self.user = user

        #purge the dataset based on the users allergies.
        self.dataset = self.remove_allergies(self.dataset, self.user)

    def remove_allergies(self, dataset: Dataset, user: user_input):
        dataset.recipes = dataset.filter_by_allergies(user.allergies)
        return dataset

    #ammount corosponds to days where each day has 3 meals.
    def generate_meal_plan(self, amount: int):
        print("hello world")


    