import copy
from models.meal_plan import MealPlan
from models.dataset import Dataset
from analyzers import random_meal_plan_generator, custom_algo_meal_plan_generator
from user_input.user_input import user_input
import pandas as pd
import random


#parameters to set for training
seed_num = 42
recipe_num = 300
num_of_datapoints = 20

def get_user_inputs():
    calorie_limit = input("Enter your calorie limit: ")
    budget = input("Enter your budget: ")
    flavor_preferences = input("Enter your flavor preferences: ")
    allergies = input("Enter your allergies: ")
    preparation_time = input("Enter your desired preparation time: ")
    cooking_utensils = input("Enter your available cooking utensils: ")
    
    return calorie_limit, budget, flavor_preferences, allergies, preparation_time, cooking_utensils


#intializes creation of training data
def init_database(file_location, dataset):
    df = pd.DataFrame()
    #add in allergies
    for text in dataset.get_ingredient_names():
        df[text] = []

    #add the columns for the recipe indexes
    df["recipe1"] = []
    df["recipe2"] = []
    df["recipe3"] = []

    #add in preferences
    for text in dataset.get_ingredient_names():
        df["preferences: " + text] = []

    #print(len(df.columns))
    #print(len(dataset.get_ingredient_names()))
    #save the dataframe as a csv to specified file location
    df.to_csv(file_location, index=False)
    #input("how doing")


#writes new entries to training data
def add_data_to_database(file_location, dataset, user: user_input, mealPlan: MealPlan):
    #load the dataset
    df = pd.read_csv(file_location)

    #build the array to add a new row to the database
    array = []
    #add allergies
    for item in dataset.get_ingredient_names():
        add_item = False
        #print(item)
        for allergy in user.allergies:
            if item == allergy:
                add_item = True
                break
        array.append(add_item)
    
    #add in the 3 recipe index
    array.append(mealPlan.recipes[0].title)
    array.append(mealPlan.recipes[1].title)
    array.append(mealPlan.recipes[2].title)

    #add preferences
    for item in dataset.get_ingredient_names():
        add_item = False
        #print(item)
        for preference in user.preferences:
            if item == preference:
                add_item = True
                break
        array.append(add_item)

    #print(len(array))
    #print(df)
    #add the array to the dataframe
    df.loc[len(df.index)] = array

    print(df)
    df.to_csv(file_location,index=False)
    #input("how doing")

def store_params_to_csv(file_location, seed, num_of_recipies, datapoints):
    df = pd.DataFrame({'seed': [seed], 'num_of_recipies': [num_of_recipies], 'datapoints': [datapoints]})
    df.to_csv(file_location, index=False)

def main():
    # Get user inputs
    # calorie_limit, budget, flavor_preferences, allergies, preparation_time, cooking_utensils = get_user_inputs()
    #record parameters
    store_params_to_csv('data/training_creation_params.csv',seed_num, recipe_num, num_of_datapoints)
    
    # preprocess data
    random.seed(seed_num)

    dataset = Dataset()
    dataset.create_recipes_from_csv(file_location="data/recipes/full.json",num_of_entries=200)

    #initialize the pandas dataframe with a column for each of the ingredients in the thing.
    init_database("data/training.csv", dataset)


    for i in range(num_of_datapoints):
        #create a copy of the dataset that will be used for the ai system
        dataCopy = copy.deepcopy(dataset)

        #get the user
        user = user_input(dataCopy.get_ingredient_names())
        #user.get_user_inputs()
        user.auto_create_user()

        #print out the inputs from the user
        print("allergies: " + str(user.allergies))
        print("preferences: " + str(user.preferences))

        analyzer = custom_algo_meal_plan_generator.customAlgoMealPlanGenerator(dataCopy, user)

        meal_plan = analyzer.generate_meal_plan(1)
        # Print the meal plan, grocery list, and estimated cost
        print("Meal Plan:")
        print(meal_plan.get_recipe_names())

        #add the meal plan to the dataset.
        add_data_to_database("data/training.csv",dataset, user, meal_plan)
        print("")


if __name__ == "__main__":
    main()
