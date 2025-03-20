def GA(X, y, count, column_count, repeatable, select_count, selection_type='roulette_wheel_selection', crossover='single_point', max_generations=100, target_accuracy=0.95, max_stagnation=50):
    initial_gen = initialize(count, column_count)
    generation = 0
    best_fitness = 0
    stagnation_counter = 0  # Track generations without improvement
    max_fitness_list = list()  
    while generation < max_generations and best_fitness < target_accuracy:
        fitness_list = list()
        
        for i in range(len(initial_gen)):
            fitness_list.append(get_fitness(initial_gen, X, i, y))
        
        current_max = max(fitness_list)
        previous_best = best_fitness
        best_fitness = max(best_fitness, current_max)

        if best_fitness == previous_best:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        
        max_fitness_list.append(current_max)
        
        if stagnation_counter >= max_stagnation:
            break
        
        if selection_type == 'roulette_wheel_selection':
            selected_ones = roulette_wheel_selection(initial_gen, fitness_list, repeatable, select_count)
        elif selection_type == 'rank_based_selection':
            selected_ones = rank_based_selection(initial_gen, fitness_list, repeatable, select_count)
        elif selection_type == 'tournament_selection':
            selected_ones = tournament_selection(initial_gen, fitness_list, repeatable, select_count)
        else:
            raise ValueError('Invalid selection')

        if crossover == 'single_point':
            offspring = single_point_crossover(selected_ones)
        elif crossover == 'two_point':
            offspring = two_point_crossover(selected_ones)
        elif crossover == 'uniform':
            offspring = uniform_crossover(selected_ones)
        else:
            raise ValueError('Invalid crossover')

        mutated_offspring = [mutate(list(child), column_count * 2) for child in offspring]
        initial_gen = mutated_offspring
        generation += 1
    
    plt.plot(max_fitness_list)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness per Generation')
    plt.show()
    
    return mutated_offspring