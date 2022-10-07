#!/usr/bin/env python3
'''
Function to get the max and min values of a list
'''

def get_max_min(list):
    max = list[0]
    min = list[0]
    for i in list:
        if i > max:
            max = i
        if i < min:
            min = i
    return max, min

def main():
    # Create a list of numbers
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Get the max and min values
    max, min = get_max_min(list)

    # Print the results
    print("Max: ", max)
    print("Min: ", min)


if __name__ == '__main__':
    main()

