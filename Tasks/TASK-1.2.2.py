"""
TASK-1.2.2: First 100 prime numbers: use for instead of while
24.09.2022
"""

def first_100_prime_nums():
    prime_list = [2,3]
    
    for i in range(5,10000):
        is_prime = True
        for j in range(3,(i//2)+1):
            if i%j == 0:
                is_prime = False
                break
        if is_prime: prime_list.append(i)
        if len(prime_list) == 100: return prime_list

print(first_100_prime_nums(),"\n---------------\n")
print("length of the list = ",len(first_100_prime_nums()))