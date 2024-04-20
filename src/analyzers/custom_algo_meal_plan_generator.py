from models.meal_plan import MealPlan, GroceryList
from models.dataset import Dataset
from user_input.user_input import user_input
import random

class customAlgoMealPlanGenerator:
    pref_weight = 1 #how much preference plays into the thing
    less_than_best_offset = 1 # how much lower off the best score do we want to add to the random selection pool

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

        plans = []
        #make a list of all combonations of 3 item lists in our dataset

        #add all the recipes to the plan list as list.
        for recipe1 in self.dataset.recipes:
            for recipe2 in self.dataset.recipes:
                for recipe3 in self.dataset.recipes:
                    plans.append([recipe1, recipe2, recipe3])


        print("scoring recipes")
        #score all the meal plans
        scores = []
        for mealPlan in plans:
            #add points for preferences
            preference_score = 0
            for recipe in mealPlan:
                #go through all the preferences
                for preference in self.user.preferences:
                    #go through each of the ingredients in a recipe
                    for ingredient in recipe.NER:
                        if ingredient.lower() == preference.lower():
                            preference_score = preference_score + self.pref_weight
                            #print("found a recipe")

            
            #add in a score for the recipe
            scores.append(0 + preference_score)
        #print(scores)
        
        print("finding max score")
        #get max score
        max_score = scores[0]
        max_score_index = 0
        print(len(scores))

        for index in range(len(scores)):
            score = scores[index]
            if score > max_score:
                max_score_index = index
                max_score = score
                #print("new best")
                #print(max_score)
        print(max_score_index)
        print(max_score)

        #get the recipes within the 
            



        




    


        


    