"""
30.10.2022
"""

class Animal():
    
    def __init__(self, num_of_feet, multi_cell, extincted, mammal):
        self.num_of_feet = num_of_feet
        self.multi_cell = multi_cell
        self.extincted = extincted
        self.mammal = mammal
        
    def can_breath(self):
        return True
        
    def can_fly(self):
        pass
    
    def can_run(self):
        pass
    
    def can_swim(self):
        pass
    
class Dinosaur(Animal):
    
    def __init__(self, num_of_feet, multi_cell, extincted, mammal, feather):
        super().__init__(num_of_feet, multi_cell, extincted, mammal)
        self.feather = feather
        
    def can_run(self):
        return True
    
class Orkinos(Animal):
    def __init__(self, num_of_feet, multi_cell  , extincted , mammal):
        super().__init__(num_of_feet, multi_cell  , extincted , mammal)     
    
    def can_run(self):
        return False
    
    def can_swim(self):
        return True

class Eagle(Animal):
    def __init__(self, num_of_feet, multi_cell  , extincted , mammal, wings):
        super().__init__(num_of_feet, multi_cell  , extincted , mammal)
        self.wings = wings
        
    def can_fly(self):
        return True        

class Gazelle(Animal):
    def __init__(self, num_of_feet, multi_cell  , extincted , mammal):
        super().__init__(num_of_feet, multi_cell  , extincted , mammal)
    
    def can_run(self):
        return True
            




    
        