def single_point_crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            p1, p2 = parents[i], parents[i + 1]
            cross_point = random.randint(1, len(p1) - 1)
            child1 = p1[:cross_point] + p2[cross_point:]
            child2 = p2[:cross_point] + p1[cross_point:]
            offspring.extend([child1, child2])
    return offspring