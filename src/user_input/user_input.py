from typing import List
import random

class user_input:
    def __init__(self, ingredients_list: List[str]):
        self.allergies = []
        self.preferences = []
        #would theoretically come from a list of all recipies
        self.ingredients = ingredients_list

    def get_allergies(self):
        return self.allergies

    def get_preferences(self):
        return self.preferences

    def get_user_inputs(self):
        print(self.ingredients)
        self.get_user_allergies()
        self.get_user_preferences()

    def auto_create_user(self):
        #generate a random length list for allergies and preferences:
        index_list = random.sample(range(len(self.ingredients)),random.randint(0,30))
        
        halfway = int(len(index_list)/2)

        #get allergies
        for index in range(halfway):
            self.allergies.append(self.ingredients[index_list[index]].lower())
        #get preferences
        for index in range(halfway, len(index_list)):
            self.preferences.append(self.ingredients[index_list[index]].lower())


    #todo: add functionality to ensure that preferences cannot be allergies
    def get_user_preferences(self):
        #reset preferences list
        self.preferences = []
        repeat = True

        #handle first case
        text = input("do you have any ingredient preferences? if none enter 'no': ")
        
        #check if the text is no
        if text.lower() == "no":
            repeat = False
        else:
            #check if the ingredient is in the list
            if self.Is_ingredient_present_in_list(text,self.ingredients):
                #reject preferences that are allergies
                if not self.Is_ingredient_present_in_list(text,self.allergies):
                    #add the preference to the list
                    self.preferences.append(text.lower())
                else:
                    print(text.lower() + " is one of your allergies. Not adding to preferences")
            else:
                print(text.lower() + " is not one of the ingredients in any of our recipes. Not adding to preferences")
        
        #get all the other preferences.
        while repeat:
            #display current preference list
            print("")
            print("your preferences are: " + str(self.preferences))
            
            #get other preferences
            text = input("do you have any additional preferences? if none enter 'no': ")
            if text.lower() == "no":
                repeat = False
            else:
                #check that it is a ingredient and not a duplicate
                if self.Is_ingredient_present_in_list(text,self.ingredients) and not self.Is_ingredient_present_in_list(text,self.preferences):
                    #check that the ingredient is not a allergy
                    if not self.Is_ingredient_present_in_list(text,self.allergies):
                        #add the thing to the preferences
                        self.preferences.append(text.lower())
                    else:
                        print(text.lower() + " is one of your allergies. Not adding to preferences")
                else:
                    print(text.lower() + " is not a ingredient in any of our recipes or is already present. we are not it appending to list")


    def get_user_allergies(self):
        # reset the allergies
        self.allergies = []
        repeat = True
        # handle the first case for allergies
        text = input("what is one of your food allergies? if none enter 'no': ")
        #check if the text is no
        if text.lower() == "no":
            repeat = False
        else:
            #check to see if the thing is in the ingredients list
            if self.Is_ingredient_present_in_list(text,self.ingredients):
                #add the ingredient to the list.
                self.allergies.append(text.lower())
            else:
                print(text.lower() + " is not a ingredient in any of our recipes. we are not it appending to list")

        # prompt the user to add additional
        while repeat:
            #display the current allergy list
            print("")
            print("your allergies are " + str(self.allergies))

            #prompt user for additional allergies
            text = input("what is another one of your food allergies? if none additional enter 'no': ")
            # check if the text is no
            if text.lower() == "no":
                repeat = False
            else:
                # check to see if the thing is in the ingredients list and if we already have it
                if self.Is_ingredient_present_in_list(text,self.ingredients) and not self.Is_ingredient_present_in_list(text,self.allergies):
                    #add the ingredient to the list.
                    self.allergies.append(text.lower())
                else:
                    print(text.lower() + " is not a ingredient in any of our recipes or is already present. we are not it appending to list")
    
    #returns true if the if the ingredient is in the list
    def Is_ingredient_present_in_list(self, ingredient: str, list: List[str]):
            inlist = False
            for item in list:
                if item.lower() == ingredient.lower():
                    inlist = True
                    # stop the for loop
                    break
            return inlist


def main():
    #for testing user interface.
    input = user_input()
    input.get_user_inputs()

if __name__ == "__main__":
    main()
