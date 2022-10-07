'''
calculator program
'''
def calculator():
    '''
    calculator function
    '''
    print("Welcome to the calculator program")
    print("Please enter the first number")
    num1 = int(input())
    print("Please enter the second number")
    num2 = int(input())
    print("Please enter the operation you would like to perform")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    operation = int(input())
    if operation == 1:
        print("The result is", num1 + num2)
    elif operation == 2:
        print("The result is", num1 - num2)
    elif operation == 3:
        print("The result is", num1 * num2)
    elif operation == 4:
        print("The result is", num1 / num2)
    else:
        print("Invalid operation")
    print("Thank you for using the calculator program")


if __name__ == '__main__':
    calculator()

