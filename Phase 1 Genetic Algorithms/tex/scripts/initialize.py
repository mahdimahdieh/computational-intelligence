def initialize(count, column_count):
    zero_array = np.zeros((1, column_count), dtype=int)
    my_set = set()

    while len(my_set) != count:
        random_numbers = random.sample(range(0, 21), column_count)
        copied_zero_array = zero_array.copy()
        for i in range(column_count):
            copied_zero_array[0][i] = random_numbers[i]
        my_set.add(tuple(copied_zero_array[0]))
    my_list = list(my_set)
    return my_set