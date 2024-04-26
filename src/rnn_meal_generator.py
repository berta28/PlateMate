#create RNN model using tensorflow

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import retrieve_training_data
from models.dataset import Dataset

class RNNMealGenerator:
    def __init__(self):
        self.df = pd.read_csv('data/test_data.csv')

def train_rnn(features, targets):
    # Convert features and targets to tensors
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    # Define hyperparameters
    input_size = features.shape[-1]
    hidden_size = 64
    output_size = targets.shape[-1]
    learning_rate = 0.001
    num_epochs = 100

    # Initialize RNN model
    model = nn.LSTM(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the RNN model
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    return model



def main():
    rnn_meal_generator = RNNMealGenerator()

    #retrieve the user and mealPlan at index 0
    tde = retrieve_training_data.training_data_extractor()
    dataset = Dataset()
    dataset.create_recipes_from_csv(file_location="data/recipes/full.json")
    ingredient_list = dataset.get_ingredient_names()
    
    # Define variables
    features = []
    target_meal_plan = []
    
    #create_training features
    for i in range(len(tde.df)):
        user, meals = tde.get_user_and_meal_plan(i)
        feature = np.zeros(len(ingredient_list))
        for ingredient in user.preferences:
            feature[ingredient_list.index(ingredient)] = 1
        for ingredient in user.allergies:
            feature[ingredient_list.index(ingredient)] = 2
        
        features.append(feature)
        target_meal_plan.append([recipe.recipe_number for recipe in meals.recipes])

    # Convert features and targets to tensors
    # features = feature_allergies
    targets = target_meal_plan
    
    # Train the RNN model
    model = train_rnn(features, targets)
    
    # Predict the next meal plan
    predictions = model(features)
    next_meal_plan = predictions.argmax(dim=1)
    print("Next Meal Plan:", next_meal_plan)




    
    

if __name__ == "__main__":
    main()