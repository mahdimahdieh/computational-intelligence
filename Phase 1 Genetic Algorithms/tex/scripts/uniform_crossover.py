def uniform_crossover(parents, swap_prob=0.5):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            p1, p2 = parents[i], parents[i + 1]
            child1, child2 = [], []
            for gene1, gene2 in zip(p1, p2):
                if random.random() < swap_prob:
                    child1.append(gene2)
                    child2.append(gene1)
                else:
                    child1.append(gene1)
                    child2.append(gene2)
            offspring.extend([''.join(child1), ''.join(child2)])
    return offspring