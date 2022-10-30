"""
30.10.2022
Lab Class oluşturunuz. PC Lab 
"""

class PC_Lab():
    
    def __init__(self, pc_total, lab_no, subject, teacher, programming_languages_in_pc = []):
        self.lab_no = lab_no
        self.subject = subject
        self.programming_languages_in_pc = programming_languages_in_pc
        self.pc_total = pc_total
        self.teacher = teacher
    
        
    def __repr__(self):
        return "The no. of the lab is : " + str(self.lab_no)
    
    def __len__(self):
        return self.pc_total
    
    def add_languages(self, languages):
        if len(set(self.programming_languages_in_pc).intersection(set(languages)))>0:
            print("There (is or are) programming language(s) that already exist(s)")
        else:
            self.programming_languages_in_pc += languages
        print("Current list : " ,self.programming_languages_in_pc)
        
    def change_sub(self, new_sub):
        self.subject = new_sub
        print("The subject has been changed successfully! New subject : ", self.subject)
        
    def update_pc_num(self, new_num):
        self.pc_total = new_num
        print(self.__len__())
    
    def delete_languages(self, languages):
        if set(languages).issubset(set(self.programming_languages_in_pc)):
            self.programming_languages_in_pc = list(set(self.programming_languages_in_pc).difference(set(languages)))
            print("Current list : ", self.programming_languages_in_pc)
        else:
            print("There are no programming languages named : ", languages)

lab_1 = PC_Lab(30, 1, "OOP", "Teacher 1")
text = """Choose one of these options : 
    1. Represent
    2. Show me how many computers there are in the lab
    3. Add programming languages
    4. Change the subject
    5. Update the number of PC
    6. Delete languages from programming language list
    7. Quit
----------------------------------------------------------
"""
            
while True:
    option = input(text)
    
    if option == "1":
        print(repr(lab_1))
    
    elif option == "2":
        print("There are ",len(lab_1), " computers in the lab.")
        
    elif option == "3":
        x = input("Type the programming languages that you want to add with ',' between them: ")
        lab_1.add_languages(x.split(","))
        
    elif option == "4":
        x = input("Enter the subject's name : ")
        lab_1.change_sub(x)
        
    elif option == "5":
        x = int(input("New number : "))
        lab_1.update_pc_num(x)
        print("There are ",len(lab_1), " computers in the lab.")
    
    elif option == "6":
        x = input("Type the programming languages that you want to delete with ',' between them: ")
        lab_1.delete_languages(x.split(","))
    
    elif option == "7":
        print("The program has been shut down.")
        break
    
    else:
        print("Invalid input. Try again.")
    
    
    
    
    
    
    
    
    
    
        
    