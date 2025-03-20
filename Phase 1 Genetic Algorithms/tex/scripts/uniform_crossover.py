def uniform_crossover(parents, swap_prob=0.5):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            p1, p2 = parents[i], parents[i + 1]
            child_size = len(p1)
            make1 = []
            make2 = []
            for g1, g2 in zip(p1, p2):
                if random.random() < swap_prob:
                    make1.append(g2)
                    make2.append(g1)
                else:
                    make1.append(g1)
                    make2.append(g2)
            make1 += p1 + p2
            make2 += p1 + p2
            set1 = set()
            set2 = set()
            for gene in make1:
                if len(set1) == child_size:
                    break
                set1.add(gene)
            for gene in make2:
                if len(set2) == child_size:
                    break
                set2.add(gene)
            offspring.append(list(set1))
            offspring.append(list(set2))
    return offspring