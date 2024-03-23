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
            inlist = False
            for item in self.ingredients:
                if item.lower() == text.lower():
                    #add the thing to the allergy list
                    self.allergies.append(text.lower())
                    inlist = True
                    #stop the for loop
                    break
            if not inlist:
                print(text.lower() + " is not a ingredient in any of our recipes. we are not it appending to list")

        # prompt the user to add additional
        while repeat:
            #display the current allergy list
            print("your allergies are " + str(self.allergies))

            #prompt user for additional allergies
            text = input("what is another one of your food allergies? if none addition enter 'no': ")
            # check if the text is no
            if text.lower() == "no":
                repeat = False
            else:
                # check to see if the thing is in the ingredients list
                inlist = False
                for item in self.ingredients:
                    if item.lower() == text.lower():
                        # add the thing to the allergy list
                        self.allergies.append(text.lower())
                        inlist = True
                        # stop the for loop
                        break
                if not inlist:
                    print(text.lower() + " is not a ingredient in any of our recipes. we are not it appending to list")

def main():
    #for testing user interface.
    input = user_input()
    input.get_user_inputs()

if __name__ == "__main__":
    main()
