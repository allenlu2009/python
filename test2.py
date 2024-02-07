# write a quick sort program

def quick_sort(arr):
    if len(arr) < 2:
        return arr

    pivot = arr[0]
    less = [i for i in arr[1:] if i <= pivot]
    greater = [i for i in arr[1:] if i > pivot]
    return quick_sort(less) + [pivot] + quick_sort(greater)


print(quick_sort([10, 5, 2, 3]))


# write a bubble sort program
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# test bubble sort program and compare with quick sort program
arr = [10, 5, 2, 3]
print(bubble_sort(arr))
print(quick_sort(arr))
print(bubble_sort(arr) == quick_sort(arr))



