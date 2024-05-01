import os
import sys
import pulp
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulp import LpProblem, PULP_CBC_CMD


class EOCGMealPlanGenerator:
    def __init__(self, dataset, user, nutritional_limits):
        self.dataset = dataset
        self.user = user
        self.recipes = self.load_and_filter_recipes()
        self.meal_types = ['breakfast', 'lunch', 'dinner']
        self.nutritional_limits_per_meal = nutritional_limits

    # Filter recipes based on user allergies
    def load_and_filter_recipes(self):
        allergens_set = set(allergen.lower() for allergen in self.user['allergies'])
        filtered_recipes = [recipe for recipe in self.dataset.recipes if not any(ing in allergens_set for ing in recipe['ingredients'])]
        return filtered_recipes
    
    # Solve the meal planning problem using linear programming
    def solve_mkp_eocg(self):
        prob = pulp.LpProblem("EOCG_Multidimensional_Knapsack", pulp.LpMaximize)
        solver = PULP_CBC_CMD(msg=0)
        x = {meal: pulp.LpVariable.dicts(meal, (i for i, _ in enumerate(self.recipes)), cat=pulp.LpBinary)
            for meal in self.meal_types}
        
        # Objective function: maximize preference coverage and diversity
        profits = {meal: [self.calculate_preference_score(recipe) for recipe in self.recipes]
                for meal in self.meal_types}
        prob += pulp.lpSum(profits[meal][i] * x[meal][i] for meal in self.meal_types for i in range(len(self.recipes)))

        # Add diversity constraints (each preference should ideally appear only once across all meals)
        for preference in self.user['preferences']:
            prob += pulp.lpSum(x[meal][i] for meal in self.meal_types for i, recipe in enumerate(self.recipes) if preference in recipe['ingredients']) <= 1

        # Nutritional constraints
        for meal in self.meal_types:
            for nutrient, limit in self.nutritional_limits_per_meal[meal].items():
                prob += pulp.lpSum([recipe['nutrient_values'].get(nutrient, 0) * x[meal][i] for i, recipe in enumerate(self.recipes)]) <= limit, f"{meal}_{nutrient}"

        prob.solve(solver)
        if pulp.LpStatus[prob.status] != "Optimal":
            print("No optimal solution found.")
            return {}
        
        selected_recipes = {meal: [self.recipes[i] for i in range(len(self.recipes)) if x[meal][i].value() == 1]
                            for meal in self.meal_types}
        meal_plan_combinations = {meal: random.choice(selected_recipes[meal]) for meal in self.meal_types if selected_recipes[meal]}
        return meal_plan_combinations


    # Calculate preference score based on user preferences
    def calculate_preference_score(self, recipe):
        score = 0
        used_preferences = set()  
        for ingredient in recipe['ingredients']:
            for preference in self.user['preferences']:
                if preference in ingredient and preference not in used_preferences:
                    score += 1  
                    used_preferences.add(preference)
        return score

    
    # Display meal plan details including nutritional values
    def display_meal_plan_details(self, meal_plan):
        print("Total Meal Plan Nutritional Values:")
        total_nutrition = {nutrient: 0 for nutrient in self.nutritional_limits_per_meal['breakfast'].keys()}  
        for meal_type, recipe in meal_plan.items():
            full_ingredients = ', '.join(f"{{{ing}}}" for ing in recipe['full_ingredients'])
            print(f"{meal_type.capitalize()}: {recipe['name']}")
            print(f"URL: {recipe['url']}")
            print(f"Full Ingredients: {full_ingredients}")
            for nutrient, value in recipe['nutrient_values'].items():
                total_nutrition[nutrient] += value  
            print("--------")
        # print("Total Nutritional Values for all meals:")
        # for nutrient, value in total_nutrition.items():
        #     print(f"  {nutrient}: {value:.2f}")

