def tournament_selection(initial_gen, gen_fitnesss, replacement, count, sample=2):
    selected_population = []

    if replacement:
        for _ in range(count):
            tournament_indices = random.choices(range(len(initial_gen)), k=sample)
            winner_index = max(tournament_indices, key=lambda i: gen_fitnesss[i])
            selected_population.append(initial_gen[winner_index])
    else:
        not_used = list(range(len(initial_gen)))
        thrown = []

        while len(not_used) >= sample and len(selected_population) < count:
            tournament_indices = random.sample(not_used, sample)
            winner_index = max(tournament_indices, key=lambda i: gen_fitnesss[i])
            selected_population.append(initial_gen[winner_index])
            for idx in tournament_indices:
                not_used.remove(idx)
                if idx != winner_index:
                    thrown.append(idx)

        pool = thrown + not_used
        while len(selected_population) < count and pool:
            if len(pool) >= sample:
                tournament_indices = random.sample(pool, sample)
            else:
                tournament_indices = random.choices(pool, k=sample)
            winner_index = max(tournament_indices, key=lambda i: gen_fitnesss[i])
            selected_population.append(initial_gen[winner_index])

    return selected_population