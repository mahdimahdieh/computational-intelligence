def roulette_wheel_selection(initialize_gen, fitness_list, repeatable, count):
    indices = list(range(len(fitness_list)))
    if repeatable:
        #repeatable
        selected_index = random.choices(indices, weights=fitness_list, k=count)
    else:
        #non-repeatable
        selected_index = np.random.choice(indices, size=count, replace=False, p=fitness_list/sum(fitness_list))
    selected_ones = list()
    initialize_gen = list(initialize_gen)
    for i in selected_index:
        selected_ones.append(initialize_gen[i])
    return selected_ones