import pandas as pd
from models.meal_plan import MealPlan, GroceryList, Recipe
from models.dataset import MealPlan,Dataset
from user_input.user_input import user_input

class training_data_extractor():
    def __init__(self,params: str = 'data/training_creation_params.csv', data: str = "data/training.csv"):
        #intialize the thing
        self.params = params
        self.data_location = data

    #get the user and the meal plan that was selected for that user
    def get_user_and_meal_plan(self, index: int = 0):
        mealList = []
        allergies = []
        preferences = []

        #get the dataframe with the training data
        df = pd.read_csv(self.data_location)

        #get the list of all the column headings
        columns_name = list(df.columns.values)
        #print(columns)

        #get the column we want to extract the data from
        row = list(df.iloc[index])
        #print(row)

        allergies_mode = True
        for i in range(len(row)):
            #check if the column names are any of the recipe names
            if columns_name[i] == "recipe1":
                mealList.append(row[i])
            elif columns_name[i] == "recipe2":
                mealList.append(row[i])
            elif columns_name[i] == "recipe3":
                mealList.append(row[i])
                allergies_mode = False
            #check what mode we are in
            elif allergies_mode:
                if(row[i]):
                    allergies.append(columns_name[i])
            elif not allergies_mode:
                if(row[i]):
                    preferences.append(columns_name[i][len("preferences: "):])

        #print(mealList)
        #print(allergies)
        #print(preferences)
        #get recipes as objects
        recipes = []
        dataset = self.get_dataset()
        recipes.append(dataset.get_recipe_by_title(mealList[0]))
        recipes.append(dataset.get_recipe_by_title(mealList[1]))
        recipes.append(dataset.get_recipe_by_title(mealList[2]))
        


        #create the mealPlan
        grocery_list = GroceryList().from_recipes(recipes)
        estimated_cost = grocery_list.estimated_cost
        mealPlan = MealPlan(recipes, grocery_list, estimated_cost)

        #create the user
        user = user_input(dataset.get_ingredient_names())
        user.allergies = allergies
        user.preferences = preferences
        return user, mealPlan

    
    def get_dataset(self):
        pdf = pd.read_csv(self.params)
        num_entries = list(pdf['num_of_recipies'])[0]
        #print(num_entries)
        dataset = Dataset()
        dataset.create_recipes_from_csv(file_location="data/recipes/full.json",num_of_entries=num_entries)
        return dataset





        

if __name__ == "__main__":
    #retrieve the user and mealPlan at index 0
    tde = training_data_extractor()
    user, meal_plan = tde.get_user_and_meal_plan(1)
    print("allergies: " + str(user.allergies))
    print("preferences: " + str(user.preferences))

    print("Meal Plan:")
    print(meal_plan.get_recipe_names())
    print("Grocery List:")
    print(meal_plan.grocery_list.get_ingredients())
    print("Estimated Cost: $", meal_plan.estimated_cost)
