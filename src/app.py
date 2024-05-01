from models.meal_plan import MealPlan
from models.dataset import Dataset
from analyzers import random_meal_plan_generator, custom_algo_meal_plan_generator
from user_input.user_input import user_input
from models.dataset_knapsack import Dataset
from analyzers.knapsack_meal_plan_generator import EOCGMealPlanGenerator
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_user_inputs():
    calorie_limit = input("Enter your calorie limit: ")
    budget = input("Enter your budget: ")
    flavor_preferences = input("Enter your flavor preferences: ")
    allergies = input("Enter your allergies: ")
    preparation_time = input("Enter your desired preparation time: ")
    cooking_utensils = input("Enter your available cooking utensils: ")
    
    return calorie_limit, budget, flavor_preferences, allergies, preparation_time, cooking_utensils


def main():
    # Get user inputs
    # calorie_limit, budget, flavor_preferences, allergies, preparation_time, cooking_utensils = get_user_inputs()
    
    # preprocess data

    dataset = Dataset()
    dataset.create_recipes_from_csv(file_location="data/recipes/full.json", num_of_entries=10000)

    #get the user
    user = user_input(dataset.get_ingredient_names())
    user.get_user_inputs()
    #user.auto_create_user()

    #print out the inputs from the user
    print(user.allergies)
    print(user.preferences)

    analyzer = custom_algo_meal_plan_generator.customAlgoMealPlanGenerator(dataset, user)

    meal_plan = analyzer.generate_meal_plan(1)

    #analyzer = random_meal_plan_generator.RandomMealPlanAnalyzer(dataset)
    #generate 5 recipes for a meal plan
    #meal_plan = analyzer.generate_meal_plan(5)
    
    # Print the meal plan, grocery list, and estimated cost
    print("Meal Plan:")
    print(meal_plan.get_recipe_names())
    print("Grocery List:")
    print(meal_plan.grocery_list.get_ingredients())
    #print("Estimated Cost: $", meal_plan.estimated_cost)
    print_total_nutrients(meal_plan)
    
    print("Input allergies:")
    allergies = input().split(',')
    print("Input preference:")
    preferences = input().split(',')

    user_knapsack = {
        'allergies': [allergy.strip().lower() for allergy in allergies],
        'preferences': [preference.strip().lower() for preference in preferences]
    }

    dataset_knapsack = Dataset()
    dataset_knapsack.create_recipes_from_csv(file_location="E:/WPI/AI/GroupProject/PlateMate/data/recipes_with_nutritional_info.json", num_of_entries=10000)

    nutritional_limits_per_meal = {
        'breakfast': {'energy': 500, 'fat': 20, 'protein': 15, 'salt': 0.7, 'saturates': 5, 'sugars': 15},
        'lunch': {'energy': 700, 'fat': 29, 'protein': 15, 'salt': 0.8, 'saturates': 7, 'sugars': 15},
        'dinner': {'energy': 800, 'fat': 29, 'protein': 25, 'salt': 0.8, 'saturates': 8, 'sugars': 25}
    }
    knapsack_generator = EOCGMealPlanGenerator(dataset_knapsack, user_knapsack, nutritional_limits_per_meal)
    knapsack_meal_plan = knapsack_generator.solve_mkp_eocg()
    if not knapsack_meal_plan:
        print("No optimal meal plan found.")
    else:
        knapsack_generator.display_meal_plan_details(knapsack_meal_plan)

def print_total_nutrients(meal_plan):
    #sum nutritional values
    total_energy = 0
    total_fat = 0
    total_protein = 0
    total_salt = 0
    total_saturates = 0
    total_sugars = 0
    for recipe in meal_plan.recipes:
        total_energy = total_energy + recipe.get_nutrient_values().get('energy')
        total_fat = total_fat + recipe.get_nutrient_values().get('fat')
        total_protein = total_protein + recipe.get_nutrient_values().get('protein')
        total_salt = total_salt + recipe.get_nutrient_values().get('salt')
        total_saturates = total_saturates + recipe.get_nutrient_values().get('saturates')
        total_sugars = total_sugars + recipe.get_nutrient_values().get('sugars')
    print("Meal Plan Total Nutritional Values:")
    print({'calories': total_energy, 'fat': total_fat, 'protein': total_protein, 'salt': total_salt, 'saturates': total_saturates, 'sugars': total_sugars})

if __name__ == "__main__":
    main()
