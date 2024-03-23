from typing import List

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
        self.get_user_allergies()
        self.get_user_preferences

    #todo: add functionality to ensure that preferences cannot be allergies
    def get_user_preferences(self):
        #reset preferences list
        self.preferences = []
        repeat = True

        #handle first case
        text = input("do you have any preferences? if none enter 'no'")

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
            print("your allergies are " + str(self.allergies))

            #prompt user for additional allergies
            text = input("what is another one of your food allergies? if none additional enter 'no': ")
            # check if the text is no
            if text.lower() == "no":
                repeat = False
            else:
                # check to see if the thing is in the ingredients list
                if self.Is_ingredient_present_in_list(text,self.ingredients):
                    #add the ingredient to the list.
                    self.allergies.append(text.lower())
                else:
                    print(text.lower() + " is not a ingredient in any of our recipes. we are not it appending to list")
    
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
