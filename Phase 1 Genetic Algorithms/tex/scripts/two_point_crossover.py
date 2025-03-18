def two_point_crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            p1, p2 = parents[i], parents[i + 1]
            point1 = random.randint(1, len(p1) - 2)
            point2 = random.randint(point1 + 1, len(p1) - 1)
            child1 = p1[:point1] + p2[point1:point2] + p1[point2:]
            child2 = p2[:point1] + p1[point1:point2] + p2[point2:]
            offspring.extend([child1, child2])
    return offspring