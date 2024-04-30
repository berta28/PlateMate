# create RNN model using tensorflow

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.meal_plan import MealPlan, Recipe, GroceryList, Ingredient
import retrieve_training_data
from analyzers import custom_algo_meal_plan_generator

from models.dataset import Dataset
from user_input.user_input import user_input


class RNNMealGenerator:
    def __init__(self):
        self.df = pd.read_csv("data/test_data.csv")

    def train_rnn(self, features, targets):
        # Convert features and targets to tensors
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Define hyperparameters
        size = features.size()
        input_size = features.shape[-1]
        hidden_size = 3
        output_size = targets.shape[-1]
        learning_rate = 0.001
        num_epochs = 100

        # Initialize RNN model
        model = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the RNN model
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs, hidden = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

        return model

    def convert_input_to_tensor(self, user):
        dataset = Dataset()
        dataset.create_recipes_from_csv(file_location="data/recipes/full.json")
        ingredient_list = dataset.get_ingredient_names()
        features = []
        feature = np.zeros(len(ingredient_list))
        for ingredient in user.preferences:
            feature[ingredient_list.index(ingredient)] = 1
        for ingredient in user.allergies:
            feature[ingredient_list.index(ingredient)] = 2

        features.append(feature)

        return torch.tensor(features, dtype=torch.float32)

    def create_model(self):
        rnn_meal_generator = RNNMealGenerator()

        # retrieve the user and mealPlan at index 0
        tde = retrieve_training_data.training_data_extractor()
        self.dataset = Dataset()
        self.dataset.create_recipes_from_csv(file_location="data/recipes/full.json")
        ingredient_list = self.dataset.get_ingredient_names()

        # Define variables
        features = []
        target_meal_plan = []

        # create_training features
        for i in range(len(tde.df)):
            user, meals = tde.get_user_and_meal_plan(i)
            feature = np.zeros(len(ingredient_list))
            for ingredient in user.preferences:
                feature[ingredient_list.index(ingredient)] = 1
            for ingredient in user.allergies:
                feature[ingredient_list.index(ingredient)] = 2

            features.append(feature)
            target_meal_plan.append([recipe.recipe_number for recipe in meals.recipes])

            # split data 80/20
        split = int(len(features) * 0.8)
        train_feature_allergies = features[:split]
        train_target_meal_plan = target_meal_plan[:split]
        test_feature = features[split:]
        test_target_meal_plan = target_meal_plan[split:]

        # Train the RNN model
        model = self.train_rnn(train_feature_allergies, train_target_meal_plan)

        # Save the model
        torch.save(model.state_dict(), "rnn_model.pth")

        # make predictions
        model.eval()
        with torch.no_grad():
            test_input = torch.tensor(test_feature, dtype=torch.float32)
            predictions, hidden = model(test_input)
            print(predictions)

        self.model = model

    def predict(self, user):
        test_input = self.convert_input_to_tensor(user)
        with torch.no_grad():
            predictions, hidden = self.model(test_input)
            # convert predictions to meal plan
            # print(predictions)
            print(predictions)
            # count the number of recipes in the dataset
            number_of_recipes = len(self.dataset.recipes)

            # multiple by probability of each recipe
            recipe_list = []
            recipes_numbers = []
            for prob in predictions:
                for pro in prob:
                    recipe_id = int(abs(pro) * number_of_recipes)
                    recipe = self.dataset.recipes[recipe_id]
                    recipes_numbers.append(recipe_id)
                    recipe_list.append(recipe)

        for recipe in recipe_list:
            print(recipe.title)

        # return MealPlan(
        #     recipes=recipe_list,
        #     grocery_list=GroceryList(ingredients=Ingredient("None")),
        #     estimated_cost=1,
        # )
        return recipes_numbers


if __name__ == "__main__":
    rnn_meal_generator = RNNMealGenerator()
    rnn_meal_generator.create_model()
    user = user_input(rnn_meal_generator.dataset.get_ingredient_names())
    input = rnn_meal_generator.convert_input_to_tensor(user)
    p = rnn_meal_generator.predict(user)
    customAlgoMealPlanGenerator = (
        custom_algo_meal_plan_generator.customAlgoMealPlanGenerator(
            rnn_meal_generator.dataset, user
        )
    )
    meal_plan = custom_algo_meal_plan_generator.internalMealPlan(index=0, mealList=p)
    score = customAlgoMealPlanGenerator.score_plan(meal_plan)

    print("Meal Plan Score: " + str(score))
