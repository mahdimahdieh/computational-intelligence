def single_point_crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            set1 = set()
            set2 = set()
            p1, p2 = parents[i], parents[i + 1]
            child_size = len(p1)
            cross_point = random.randint(1, len(p1) - 1)
            make1 = p1[:cross_point] + p2[cross_point:] + p1[cross_point:] + p2[:cross_point]
            make2 = p2[:cross_point] + p1[cross_point:] + p2[cross_point:] + p1[:cross_point]
            for j in make1:
                if len(set1) == child_size:
                    break
                set1.add(j)
            for j in make2:
                if len(set2) == child_size:
                    break
                set2.add(j)
            child1 = list(set1)
            child2 = list(set2)
            offspring.extend([child1, child2])
    return offspring