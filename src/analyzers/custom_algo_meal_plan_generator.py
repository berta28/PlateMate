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

    pref_weight = 1000 #how much preference plays into the thing
    less_than_best_offset = 0.1 # how much lower off the best score do we want to add to the random selection pool

    recipe_repeat_weight = -1000 #how much weight a repeated recipe in a meal plan counts towards the favorablity of the system.

    norm_energy_score = 2000
    max_energy_deviation = 200 #max calories off before we start taking away points from meal plan
    energy_score_weight = -1 #how much the energy score influences the score. should always be a negative number

    norm_fat_score = 78
    max_fat_deviation = 20
    fat_score_weight = -0.5

    norm_protein_score = 50
    max_protein_deviation = 5
    protein_score_weight = -1

    norm_salt_score = 2.3
    max_salt_deviation = 0.1
    salt_score_weight = -0.5

    norm_saturates_score = 20
    max_saturates_deviation = 3
    saturates_score_weight = -0.5

    norm_sugars_score = 50
    max_sugars_deviation = 10
    sugars_score_weight = -0.5

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
        #the list of recipes is a list of the indexes to the recipes.
        for recipe1 in range(len(self.dataset.recipes)):
            for recipe2 in range(len(self.dataset.recipes)):
                for recipe3 in range(len(self.dataset.recipes)):
                    plans.append(internalMealPlan(index, [recipe1, recipe2, recipe3]))
                    index = index + 1
        return plans
    

    #goes from indexes to recipes
    def get_recipe_by_indexes(self, indexes):
        retList = []
        for index in indexes:
            retList.append(self.dataset.recipes[index])
        return retList


    def score_plans(self, plans):
        #create global varible to return the dictionary of data keyed by index of the meal plan.
        self.scores = {}
        thread = 1
        total_threads = math.ceil(len(plans)/self.max_list_size)


        #determine whether or not to run in single or multithreaded mode
        if len(plans)/self.max_list_size <= 1:
            #single threaded
            self.scores.update(self.score_plans_helper(plans))

        else:
        #multi threaded
        #split the plans with each thread responsible for
            print("splitting scoring workload")
            #plansList = np.array_split(plans, math.ceil(len(plans)/self.max_list_size))
            plansList = np.array_split(plans, math.ceil(self.max_thread_workers))
            print("preforming scoring, starting threads")
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_thread_workers)
            for result in pool.map(self.score_plans_helper, plansList):
                self.scores.update(result)
                print("done with job: " + str(thread) + " of " + str(self.max_thread_workers))
                thread = thread + 1


        
        return self.scores
    
    def score_plans_helper(self, plans):

        result_list = {}
        for index in range(len(plans)):
            mealList = plans[index]
            result_list[mealList.index] = self.score_plan(mealList)
        return result_list
    
    #scores induvidual plans
    def score_plan(self, plan: internalMealPlan):
        #add points for preferences
        mealPlan = self.get_recipe_by_indexes(plan.mealList)
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
        #sum nutritional values
        total_energy = 0
        total_fat = 0
        total_protein = 0
        total_salt = 0
        total_saturates = 0
        total_sugars = 0
        for recipe in mealPlan:
            total_energy = total_energy + recipe.get_nutrient_values().get('energy')
            total_fat = total_fat + recipe.get_nutrient_values().get('fat')
            total_protein = total_protein + recipe.get_nutrient_values().get('protein')
            total_salt = total_salt + recipe.get_nutrient_values().get('salt')
            total_saturates = total_saturates + recipe.get_nutrient_values().get('saturates')
            total_sugars = total_sugars + recipe.get_nutrient_values().get('sugars')



        #score energy
        energy_score = self.score_within_range(total_energy, self.norm_energy_score, self.max_energy_deviation, self.energy_score_weight)
        
        #print("total energy: " + str(total_energy))
        #print("energy score: " + str(energy_score))

        #run calcs for fats
        fat_score = self.score_within_range(total_fat, self.norm_fat_score, self.max_fat_deviation, self.fat_score_weight)
        #print("total_fat: " + str(total_fat))
        #print("fat score: " + str(fat_score))

        #score protein
        protein_score = self.score_within_range(total_protein, self.norm_protein_score, self.max_protein_deviation, self.protein_score_weight)

        #score salt
        salt_score = self.score_within_range(total_salt, self.norm_salt_score, self.max_salt_deviation, self.salt_score_weight)

        #score saturates
        saturates_score = self.score_within_range(total_saturates, self.norm_saturates_score, self.max_saturates_deviation, self.saturates_score_weight)
        
        #score sugars
        sugars_score = self.score_within_range(total_sugars, self.norm_sugars_score, self.max_sugars_deviation, self.sugars_score_weight)

        #add a score based on repeats
        a = mealPlan[0].recipe_number
        b = mealPlan[1].recipe_number
        c = mealPlan[2].recipe_number
        #2 of 3
        if((a == b and b != c) or (b == c and a !=c ) or (a == c and b != c)):
            repeat_score = self.recipe_repeat_weight
        elif (a == b and b == c):
            repeat_score = self.recipe_repeat_weight * 2
        else:
            repeat_score = 0
            

        #add in a score for the recipe
        return preference_score + energy_score + fat_score + protein_score + salt_score + saturates_score + sugars_score + repeat_score

    def score_within_range(self, value, target, threshold, multiplier):
        if(value > target + threshold):
            score = (value - (target + threshold)) * multiplier
        elif(value < target - threshold):
            score = ((target - threshold) - value) * multiplier
        else:
            score = 0
        return score




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
            recipes = self.get_recipe_by_indexes(desiredPlans[random.randrange(0,len(desiredPlans))].mealList)
            grocery_list = GroceryList().from_recipes(recipes)
            estimated_cost = grocery_list.estimated_cost
            return MealPlan(recipes, grocery_list, estimated_cost)
        if amount == -1:
            #to get a single days meal plan as its index range
            return desiredPlans[random.randrange(0,len(desiredPlans))].mealList
        

            



        




    


        


    