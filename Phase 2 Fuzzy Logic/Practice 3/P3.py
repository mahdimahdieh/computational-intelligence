import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


physical_fitness = ctrl.Antecedent(np.arange(0, 13, 1), 'physical_fitness')
energy = ctrl.Antecedent(np.arange(0, 11, 1), 'energy')
sports_goal = ctrl.Antecedent(np.arange(0, 11, 1), 'sports_goal')
age = ctrl.Antecedent(np.arange(15, 81, 1),'age')
weight = ctrl.Antecedent(np.arange(40, 121, 1), 'weight')

exercise_intensity = ctrl.Consequent(np.arange(0, 101, 1), 'exercise_intensity')
exercise_time = ctrl.Consequent(np.arange(0, 91, 1), 'exercise_time')



physical_fitness['beginner'] = fuzz.trapmf(physical_fitness.universe, (0,0,3,6))
physical_fitness['intermediate']= fuzz.gaussmf(physical_fitness.universe, 6, 1.5)
physical_fitness['advanced']= fuzz.trapmf(physical_fitness.universe, (9, 12, 12, 12))

energy['low'] = fuzz.trapmf(energy.universe, (0, 0, 3, 5))
energy['medium'] = fuzz.gaussmf(energy.universe, 5, 1.5)
energy['high'] = fuzz.trapmf(energy.universe, (7, 9, 10, 10))

sports_goal['weight loss'] = fuzz.trapmf(sports_goal.universe, (0, 0, 3, 5))
sports_goal['muscle gain'] = fuzz.gaussmf(sports_goal.universe, 5, 1.5)
sports_goal['general fitness'] = fuzz.trapmf(sports_goal.universe, (6, 8, 10, 10))

age['young'] = fuzz.trapmf(age.universe, (15, 15, 30, 40))
age['middle-aged'] = fuzz.gaussmf(age.universe, 45, 10)
age['elderly'] = fuzz.trapmf(age.universe, (60, 70, 80, 80))

weight['underweight'] = fuzz.trapmf(weight.universe, (40, 40, 50, 60))
weight['normal'] = fuzz.gaussmf(weight.universe, 70, 10)
weight['overweight'] = fuzz.trapmf(weight.universe, (80, 90, 120, 120))

exercise_intensity['low'] = fuzz.trapmf(exercise_intensity.universe, (0, 0, 30, 50))
exercise_intensity['medium'] = fuzz.gaussmf(exercise_intensity.universe, 50, 10)
exercise_intensity['high'] = fuzz.trapmf(exercise_intensity.universe, (60, 70, 100, 100))

exercise_time['short'] = fuzz.trapmf(exercise_time.universe, (0, 0, 20, 40))
exercise_time['medium'] = fuzz.gaussmf(exercise_time.universe, 40, 10)
exercise_time['long'] = fuzz.trapmf(exercise_time.universe, (50, 60, 90, 90))


def show(membership_func, name):
    plt.figure(figsize=(8, 5))  # ایجاد یک شکل جدید
    membership_functions = membership_func.terms.items()  # دریافت توابع عضویت و نام آنها
    for term_name, term in membership_functions:
        plt.plot(membership_func.universe, term.mf, label=term_name, linewidth=1.5)  # رسم هر تابع عضویت
    plt.title(name + " Membership Functions")
    plt.xlabel(name)
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

show(physical_fitness, 'physical fitness')
show(energy, 'energy')
show(sports_goal, 'sports goal')
show(age, 'age')
show(weight, 'weight')

show(exercise_intensity, 'exercise intensity')
show(exercise_time, 'exercise time')



# Rules:
rule1 = ctrl.Rule(antecedent=(physical_fitness['beginner'] & energy['low']),
                  consequent=(exercise_intensity['low'], exercise_time['short']),
                  label='rule1: beginner_low_energy')

rule2 = ctrl.Rule(antecedent=(physical_fitness['beginner'] & (energy['medium'] | energy['high'])),
                  consequent=exercise_intensity['low'],
                  label='rule2: beginner_higher_energy_intensity_low')

rule3 = ctrl.Rule(antecedent=(physical_fitness['beginner'] & (energy['medium'] | energy['high'])),
                  consequent=exercise_intensity['medium'],
                  label='rule3: beginner_higher_energy_intensity_medium')

rule4 = ctrl.Rule(antecedent=(physical_fitness['beginner'] & (energy['medium'] | energy['high'])),
                  consequent=exercise_time['medium'],
                  label='rule4: beginner_higher_energy_time')

rule5 = ctrl.Rule(antecedent=(physical_fitness['intermediate'] & energy['high']),
                  consequent=exercise_intensity['medium'],
                  label='rule5: intermediate_high_energy_intensity_medium')

rule6 = ctrl.Rule(antecedent=(physical_fitness['intermediate'] & energy['high']),
                  consequent=exercise_intensity['high'],
                  label='rule6: intermediate_high_energy_intensity_high')

rule7 = ctrl.Rule(antecedent=(physical_fitness['intermediate'] & energy['high']),
                  consequent=exercise_time['medium'],
                  label='rule7: intermediate_high_energy_time_medium')

rule8 = ctrl.Rule(antecedent=(physical_fitness['intermediate'] & energy['high']),
                  consequent=exercise_time['long'],
                  label='rule8: intermediate_high_energy_time_long')

rule9 = ctrl.Rule(antecedent=(sports_goal['weight loss'] & age['elderly']),
                  consequent=exercise_intensity['low'],
                  label='rule9: weight_loss_elderly_intensity_low')

rule10 = ctrl.Rule(antecedent=(sports_goal['weight loss'] & age['elderly']),
                   consequent=exercise_intensity['medium'],
                   label='rule10: weight_loss_elderly_intensity_medium')

rule11 = ctrl.Rule(antecedent=(sports_goal['weight loss'] & age['elderly']),
                   consequent=exercise_time['short'],
                   label='rule11: weight_loss_elderly_time')

rule12 = ctrl.Rule(antecedent=(sports_goal['muscle gain'] & age['young']),
                   consequent=exercise_intensity['high'],
                   label='rule12: muscle_gain_young_intensity_high')

rule13 = ctrl.Rule(antecedent=(sports_goal['muscle gain'] & age['young']),
                   consequent=exercise_time['medium'],
                   label='rule13: muscle_gain_young_time_medium')

rule14 = ctrl.Rule(antecedent=(sports_goal['muscle gain'] & age['young']),
                   consequent=exercise_time['long'],
                   label='rule14: muscle_gain_young_time_long')

rule15 = ctrl.Rule(antecedent=(sports_goal['general fitness'] & physical_fitness['intermediate'] & energy['medium']),
                   consequent=(exercise_intensity['medium'], exercise_time['medium']),
                   label='rule15: general_fitness')

rule16 = ctrl.Rule(antecedent=(weight['overweight'] & sports_goal['weight loss']),
                   consequent=exercise_intensity['low'],
                   label='rule16: overweight_weight_loss_intensity_low')

rule17 = ctrl.Rule(antecedent=(weight['overweight'] & sports_goal['weight loss']),
                   consequent=exercise_intensity['medium'],
                   label='rule17: overweight_weight_loss_intensity_medium')

rule18 = ctrl.Rule(antecedent=(weight['overweight'] & sports_goal['weight loss']),
                   consequent=exercise_time['short'],
                   label='rule18: overweight_weight_loss_time')

rule19 = ctrl.Rule(antecedent=(physical_fitness['intermediate'] & energy['low']),
                   consequent=(exercise_intensity['low'], exercise_time['short']),
                   label='rule19: intermediate_low_energy')

rule20 = ctrl.Rule(antecedent=(physical_fitness['advanced'] & energy['high']),
                   consequent=(exercise_intensity['high'], exercise_time['long']),
                   label='rule20: advanced_high_energy')

rule21 = ctrl.Rule(antecedent=(age['young'] & sports_goal['general fitness']),
                   consequent=(exercise_intensity['medium'], exercise_time['medium']),
                   label='rule21: young_general_fitness')

rule22 = ctrl.Rule(antecedent=(sports_goal['muscle gain'] & age['middle-aged']),
                   consequent=exercise_intensity['medium'],
                   label='rule22: middle_aged_muscle_gain_intensity_medium')

rule23 = ctrl.Rule(antecedent=(sports_goal['muscle gain'] & age['middle-aged']),
                   consequent=exercise_intensity['high'],
                   label='rule23: middle_aged_muscle_gain_intensity_high')

rule24 = ctrl.Rule(antecedent=(sports_goal['muscle gain'] & age['middle-aged']),
                   consequent=exercise_time['short'],
                   label='rule24: middle_aged_muscle_gain_time_short')

rule25 = ctrl.Rule(antecedent=(weight['overweight'] & energy['high'] & sports_goal['general fitness']),
                   consequent=exercise_intensity['medium'],
                   label='rule25: overweight_high_energy_general_fitness_intensity_medium')

rule26 = ctrl.Rule(antecedent=(weight['overweight'] & energy['high'] & sports_goal['general fitness']),
                   consequent=exercise_intensity['high'],
                   label='rule26: overweight_high_energy_general_fitness_intensity_high')

rule27 = ctrl.Rule(antecedent=(weight['overweight'] & energy['high'] & sports_goal['general fitness']),
                   consequent=exercise_time['medium'],
                   label='rule27: overweight_high_energy_general_fitness_time_medium')



# Fuzzy Control System:
all_rules = [
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8,
    rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16,
    rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24,
    rule25, rule26, rule27
]


exercise_ctrl = ctrl.ControlSystem(all_rules)
exercise_simulation = ctrl.ControlSystemSimulation(exercise_ctrl)


# samples:
exercise_simulation.input['physical_fitness'] = 7
exercise_simulation.input['energy'] = 8
exercise_simulation.input['sports_goal'] = 6
exercise_simulation.input['age'] = 30
exercise_simulation.input['weight'] = 75

exercise_simulation.compute()

#crisp:
print("Intensity:", exercise_simulation.output['exercise_intensity'])
print("Time:", exercise_simulation.output['exercise_time'])


exercise_intensity.view(sim = exercise_simulation)
exercise_time.view(sim = exercise_simulation)