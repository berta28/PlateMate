from models.meal_plan import MealPlan
def get_user_inputs():
    calorie_limit = input("Enter your calorie limit: ")
    budget = input("Enter your budget: ")
    flavor_preferences = input("Enter your flavor preferences: ")
    allergies = input("Enter your allergies: ")
    preparation_time = input("Enter your desired preparation time: ")
    cooking_utensils = input("Enter your available cooking utensils: ")
    
    return calorie_limit, budget, flavor_preferences, allergies, preparation_time, cooking_utensils

def generate_meal_plan(calorie_limit, budget, flavor_preferences, allergies, preparation_time, cooking_utensils):
    # Your logic to generate a meal plan goes here
    # This could involve fetching recipes, calculating costs, and creating a grocery list
    
    # Return the generated meal plan, grocery list, and estimated cost
    return MealPlan

def main():
    # Get user inputs
    calorie_limit, budget, flavor_preferences, allergies, preparation_time, cooking_utensils = get_user_inputs()
    
    # Generate meal plan
    meal_plan = generate_meal_plan(calorie_limit, budget, flavor_preferences, allergies, preparation_time, cooking_utensils)
    
    # Print the meal plan, grocery list, and estimated cost
    print("Meal Plan:")
    print(meal_plan.recipes)
    print("Grocery List:")
    print(meal_plan.grocery_list)
    print("Estimated Cost: $", meal_plan.estimated_cost)

if __name__ == "__main__":
    main()
