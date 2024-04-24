from models.meal_plan import MealPlan, GroceryList, Recipe
from models.dataset import Dataset
from user_input.user_input import user_input
import random
import concurrent.futures
import time
import numpy as np
import math

class internalMealPlan():
    def __init__(self, index : int, mealList:list):
        self.index = index
        self.mealList = mealList

class customAlgoMealPlanGenerator:
    max_thread_workers = 24 #number of threads to spawn during multithreaded processies
    max_list_size = 100000 #length of lists before splitting

    pref_weight = 1 #how much preference plays into the thing
    less_than_best_offset = 1 # how much lower off the best score do we want to add to the random selection pool

    norm_energy_score = 20000
    max_energy_deviation = 2000 #max calories off before we start taking away points from meal plan
    energy_score_weight = -1 #how much the energy score influences the score. should always be a negative number

    def __init__(self, dataset : Dataset, user : user_input):
        self.dataset = dataset
        self.user = user

        #purge the dataset based on the users allergies.
        self.dataset = self.remove_allergies(self.dataset, self.user)

    def remove_allergies(self, dataset: Dataset, user: user_input):
        dataset.recipes = dataset.filter_by_allergies(user.allergies)
        return dataset
    
    def generate_all_plans(self):
        plans = []
        index = 0
        #add all the recipes to the plan list as list.
        for recipe1 in self.dataset.recipes:
            for recipe2 in self.dataset.recipes:
                for recipe3 in self.dataset.recipes:
                    plans.append(internalMealPlan(index, [recipe1, recipe2, recipe3]))
                    index = index + 1
        return plans

    def score_plans(self, plans):
        #create global varible to return the dictionary of data keyed by index of the meal plan.
        self.scores = {}
        self.thread = 0

        # print(time.gmtime())
        #single threaded
        self.scores.update(self.score_plans_helper(plans))

        # print(time.gmtime())

        #multi threaded
        #split the plans with each thread responsible for 1,000,000 entries
        # print("splitting scoring workload")
        # plansList = np.array_split(plans, math.ceil(len(plans)/self.max_list_size))
        # print("preforming scoring")
        # pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_workers)
        # for result in pool.map(self.score_plans_helper, plansList):
        #     self.scores.update(result)

        # print(time.gmtime())
        
        return self.scores
    
    def score_plans_helper(self, plans):
        print("running thread: " + str(self.thread))
        self.thread = self.thread + 1
        result_list = {}
        for index in range(len(plans)):
            mealList = plans[index]
            result_list[mealList.index] = self.score_plan(mealList)
        return result_list
    
    #scores induvidual plans
    def score_plan(self, plan: internalMealPlan):
        #add points for preferences
        mealPlan = plan.mealList
        index = plan.index
        #print("scoring plan: " + str(index))
        preference_score = 0
        for recipe in mealPlan:
            #go through all the preferences
            for preference in self.user.preferences:
                #go through each of the ingredients in a recipe
                for ingredient in recipe.NER:
                    if ingredient.lower() == preference.lower():
                        preference_score = preference_score + self.pref_weight
                        #print("found a recipe")

        #add points for total calories.
        energy_score = 0
        #add up the total amount of energy
        total_energy = 0
        for recipe in mealPlan:
            total_energy = total_energy + recipe.get_nutrient_values().get('energy')
        
        #if it is out of bounds
        if(total_energy > self.norm_energy_score + self.max_energy_deviation):
            energy_score = (total_energy - (self.norm_energy_score + self.max_energy_deviation)) * self.energy_score_weight
        elif(total_energy < self.norm_energy_score - self.max_energy_deviation):
            energy_score = ((self.norm_energy_score - self.max_energy_deviation) - total_energy ) * self.energy_score_weight
        else:
            energy_score = 0
        
        #print("total energy: " + str(total_energy))
        #print("energy score: " + str(energy_score))

        #add in a score for the recipe
        return preference_score + energy_score
        #self.scores[index] = preference_score + energy_score




    #ammount corosponds to days where each day has 3 meals.
    def generate_meal_plan(self, amount: int):
        print("number of recipes: " + str(len(self.dataset.recipes)))
        print("making all possible plans")
        #make a list of all combonations of 3 item lists in our dataset
        plans = self.generate_all_plans()
        print("number of plans: " + str(len(plans)))

        print("scoring recipes")
        #score all the meal plans
        scores = self.score_plans(plans)

        #print(scores)
        
        print("finding max score")
        #get max score
        max_score = scores[0]
        max_score_index = 0
        print("number of plans: " + str(len(plans)))
        print("number of scores: " + str(len(scores)))

        for index in range(len(plans)):
            score = scores[index]
            if score > max_score:
                max_score_index = index
                max_score = score
                #print("new best")
                #print(max_score)
        print("max score index: " + str(max_score_index))
        print("max score: " + str(max_score))

        print("getting desireable plans")
        #get the recipes plans within the threshold and add them to a desired list
        desiredPlans = []
        for index in range(len(scores)):
            if scores[index] + self.less_than_best_offset >= max_score:
                desiredPlans.append(plans[index])
                #print(index)
                #print(scores[index])
                #print("")
        
        print("selecting Recipe plan")
        if amount == 1:
            recipes = desiredPlans[random.randrange(0,len(desiredPlans))].mealList
            grocery_list = GroceryList().from_recipes(recipes)
            estimated_cost = grocery_list.estimated_cost
            return MealPlan(recipes, grocery_list, estimated_cost)
        

            



        




    


        


    