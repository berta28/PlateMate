
from models.dataset_knapsack import Dataset
from analyzers.knapsack_meal_plan_generator import EOCGMealPlanGenerator
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    
    print("Input allergies:")
    allergies = input().split(',')
    print("Input preference:")
    preferences = input().split(',')

    user = {
        'allergies': [allergy.strip().lower() for allergy in allergies],
        'preferences': [preference.strip().lower() for preference in preferences]
    }

    dataset = Dataset()
    dataset.create_recipes_from_csv(file_location="E:/WPI/AI/GroupProject/PlateMate/data/recipes_with_nutritional_info.json", num_of_entries=10000)  # Load recipes
    
    if not dataset.recipes:
        print("Empty csv file or not correctly loaded.")
        return

    nutritional_limits_per_meal = {
        'breakfast': {'energy': 500, 'fat': 20, 'protein': 15, 'salt': 0.7, 'saturates': 5, 'sugars': 15},
        'lunch': {'energy': 700, 'fat': 29, 'protein': 15, 'salt': 0.8, 'saturates': 7, 'sugars': 15},
        'dinner': {'energy': 800, 'fat': 29, 'protein': 25, 'salt': 0.8, 'saturates': 8, 'sugars': 25}
    }

    generator = EOCGMealPlanGenerator(dataset, user,nutritional_limits_per_meal)
    optimal_meal_plans = generator.solve_mkp_eocg()

    if not optimal_meal_plans:
        print("No optimal meal plan found.")
    else:
        generator.display_meal_plan_details(optimal_meal_plans)

if __name__ == "__main__":
    main()