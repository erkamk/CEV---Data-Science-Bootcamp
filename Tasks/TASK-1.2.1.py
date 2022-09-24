"""
Prime number function: use while instead of for
24.09.2022
"""

def testPrime(number):

    if number in (2,3,4):
        if number == 4:
            return False
        return True
    
    if number <= 1:
        return False
    
    var = 5
    while var < number//2:
        if number%var == 0:
             return False
        var +=1
        
    return True


print(testPrime(2))
print(testPrime(3))
print(testPrime(4))
print(testPrime(5))
print(testPrime(6))
print(testPrime(17))
print(testPrime(111))
print(testPrime(123))
print(testPrime(1000))
print(testPrime(23))
print(testPrime(-5))
print(testPrime(0))
