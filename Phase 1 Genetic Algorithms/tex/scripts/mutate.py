def mutate(individual, max_value):
    index = random.randint(0, len(individual) - 1)
    new_value = random.randint(1, max_value)
    while new_value in individual:
        new_value = random.randint(1, max_value)
    individual[index] = new_value
    return individual