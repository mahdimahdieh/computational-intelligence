import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


def define_variables():
    # Define fuzzy input and output variables and their membership functions
    physical_fitness = ctrl.Antecedent(np.arange(0, 101, 1), 'physical_fitness')
    energy = ctrl.Antecedent(np.arange(0, 101, 1), 'energy')
    sports_goal = ctrl.Antecedent(np.arange(0, 3, 1), 'sports_goal')  # 0: weight_loss, 1: muscle_gain, 2: general_fitness
    age = ctrl.Antecedent(np.arange(15, 81, 1), 'age')  # used only for post-processing
    weight = ctrl.Antecedent(np.arange(40, 121, 1), 'weight')

    exercise_intensity = ctrl.Consequent(np.arange(0, 101, 1), 'exercise_intensity')
    exercise_time = ctrl.Consequent(np.arange(0, 121, 1), 'exercise_time')

    physical_fitness['beginner'] = fuzz.trimf(physical_fitness.universe, [0, 0, 50])
    physical_fitness['intermediate'] = fuzz.trimf(physical_fitness.universe, [25, 50, 75])
    physical_fitness['advanced'] = fuzz.trimf(physical_fitness.universe, [50, 100, 100])

    energy['low'] = fuzz.trimf(energy.universe, [0, 0, 50])
    energy['medium'] = fuzz.trimf(energy.universe, [25, 50, 75])
    energy['high'] = fuzz.trimf(energy.universe, [50, 100, 100])

    sports_goal['weight_loss'] = fuzz.trimf(sports_goal.universe, [0, 0, 1])
    sports_goal['muscle_gain'] = fuzz.trimf(sports_goal.universe, [0.5, 1, 1.5])
    sports_goal['general_fitness'] = fuzz.trimf(sports_goal.universe, [1, 2, 2])

    age['young'] = fuzz.trimf(age.universe, [15, 15, 35])
    age['middle'] = fuzz.trimf(age.universe, [25, 45, 65])
    age['elderly'] = fuzz.trimf(age.universe, [55, 80, 80])

    weight['underweight'] = fuzz.trimf(weight.universe, [40, 40, 60])
    weight['normal'] = fuzz.trimf(weight.universe, [55, 70, 85])
    weight['overweight'] = fuzz.trimf(weight.universe, [75, 120, 120])

    exercise_intensity['low'] = fuzz.trimf(exercise_intensity.universe, [0, 0, 50])
    exercise_intensity['medium'] = fuzz.trimf(exercise_intensity.universe, [25, 50, 75])
    exercise_intensity['high'] = fuzz.trimf(exercise_intensity.universe, [50, 100, 100])

    exercise_time['short'] = fuzz.trimf(exercise_time.universe, [0, 0, 40])
    exercise_time['medium'] = fuzz.trimf(exercise_time.universe, [20, 60, 100])
    exercise_time['long'] = fuzz.trimf(exercise_time.universe, [80, 120, 120])

    return physical_fitness, energy, sports_goal, age, weight, exercise_intensity, exercise_time


def define_rules(pf, en, sg, wt, ei, et):
    # Define fuzzy rules
    rules = [ctrl.Rule(pf['beginner'] & en['low'], (ei['low'], et['short']), label='bgn_low'),
             ctrl.Rule(pf['beginner'] & en['medium'], (ei['low'], et['medium']), label='bgn_med'),
             ctrl.Rule(pf['intermediate'] & en['high'], (ei['medium'], et['medium']), label='int_high'),
             ctrl.Rule(pf['advanced'] & en['high'], (ei['high'], et['long']), label='adv_high'),
             ctrl.Rule(sg['weight_loss'] & en['high'], (ei['medium'], et['long']), label='wloss_highE'),
             ctrl.Rule(wt['overweight'] & sg['weight_loss'], et['long'], label='ow_wloss_time'),
             ctrl.Rule(sg['muscle_gain'] & pf['advanced'], ei['high'], label='mgain_adv_int')]

    # Rule 1:
    # If physical fitness is beginner and energy is low,
    # then exercise intensity should be low and duration should be short.

    # Rule 2:
    # If physical fitness is beginner and energy is medium,
    # then exercise intensity should be low and duration should be medium.

    # Rule 3:
    # If physical fitness is intermediate and energy is high,
    # then exercise intensity should be medium and duration should be medium.

    # Rule 4:
    # If physical fitness is advanced and energy is high,
    # then exercise intensity should be high and duration should be long.

    # Rule 5:
    # If sports goal is weight loss and energy is high,
    # then exercise intensity should be medium and duration should be long.

    # Rule 6:
    # If weight is overweight and sports goal is weight loss,
    # then exercise duration should be long.

    # Rule 7:
    # If sports goal is muscle gain and physical fitness is advanced,
    # then exercise intensity should be high.

    return rules


def build_and_run_system(inputs, rules):
    # Build and run fuzzy inference system
    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    for var in ['physical_fitness', 'energy', 'sports_goal', 'weight']:
        sim.input[var] = inputs[var]
    sim.compute()

    intensity = sim.output['exercise_intensity']
    duration = sim.output['exercise_time']

    if inputs['age'] >= 60:
        intensity *= 0.7

    return intensity, duration, sim


def plot_memberships_and_results(pf, en, sg, ag, wt, ei, et, sim=None):
    # Plot membership functions and crisp outputs
    for var in [pf, en, sg, ag, wt, ei, et]:
        var.view()
        plt.savefig(f'{var.label}_mf.png')

    if sim:
        ei.view(sim=sim)
        plt.plot([sim.output['exercise_intensity']] * 2, [0, 1], 'k--')
        plt.savefig('intensity_result.png')

        et.view(sim=sim)
        plt.plot([sim.output['exercise_time']] * 2, [0, 1], 'k--')
        plt.savefig('time_result.png')


def main():
    pf, en, sg, ag, wt, ei, et = define_variables()
    rules = define_rules(pf, en, sg, wt, ei, et)

    pref = input("Do you prefer more intense workouts? (y/n): ")
    pref_flag = (pref.lower() == 'y')

    test_samples = [
        {'physical_fitness': 20, 'energy': 30, 'sports_goal': 0, 'age': 25, 'weight': 65},
        {'physical_fitness': 70, 'energy': 80, 'sports_goal': 1, 'age': 30, 'weight': 75},
        {'physical_fitness': 40, 'energy': 50, 'sports_goal': 0, 'age': 65, 'weight': 80},
        {'physical_fitness': 90, 'energy': 90, 'sports_goal': 2, 'age': 40, 'weight': 60},
        {'physical_fitness': 30, 'energy': 20, 'sports_goal': 0, 'age': 70, 'weight': 90},
    ]

    for test in test_samples:
        intensity, duration, sim = build_and_run_system(test, rules)
        if pref_flag:
            intensity = min(intensity * 1.2, 100)
        print(f"Input: {test}\n → Intensity: {intensity:.1f}, Duration: {duration:.1f} minutes\n")

    _, _, last_sim = build_and_run_system(test_samples[-1], rules)
    plot_memberships_and_results(pf, en, sg, ag, wt, ei, et, sim=last_sim)


if __name__ == '__main__':
    main()