import unittest

'''
Function to get the max and min values of a list
'''

def get_max_min(lst):
    if len(lst) == 0:
        return None, None
    max_val = lst[0]
    min_val = lst[0]
    for i in lst:
        if i > max_val:
            max_val = i
        if i < min_val:
            min_val = i
    return max_val, min_val



class GetMaxMinTest(unittest.TestCase):

    def test_get_max_min(self):
        # 測試正常輸入情況
        self.assertEqual(get_max_min([1, 2, 3, 4, 5]), (5, 1))
        self.assertEqual(get_max_min([-1, -2, -3, -4, -5]), (-1, -5))
        self.assertEqual(get_max_min([5, 5, 5, 5, 5]), (5, 5))
        
    def test_get_max_min_empty_list(self):
        # 測試空列表的情況
        self.assertEqual(get_max_min([]), (None, None))
        
    def test_get_max_min_single_element(self):
        # 測試只有一個元素的情況
        self.assertEqual(get_max_min([100]), (100, 100))
        
    def test_get_max_min_duplicate_elements(self):
        # 測試存在重複元素的情況
        self.assertEqual(get_max_min([1, 3, 5, 1, 3]), (5, 1))
        
if __name__ == '__main__':
    unittest.main()



#def main():
#    # Create a list of numbers
#    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Get the max and min values
#    max, min = get_max_min(list)

    # Print the results
#    print("Max: ", max)
#    print("Min: ", min)


#if __name__ == '__main__':
#    main()

