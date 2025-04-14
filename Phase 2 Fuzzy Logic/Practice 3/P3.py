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
rule1 = ctrl.Rule(
                  antecedent=(physical_fitness['beginner'] & energy['low']),
                  consequent=(exercise_intensity['low'], exercise_time['short']),
                  label='beginner_low_energy')

# rule2 = ctrl.Rule(antecedent=(physical_fitness['beginner'] & (energy['medium'] | energy['high'])),
#                   consequent=(exercise_intensity['low'] | exercise_intensity['medium'],
#                               exercise_time['medium']),
#                   label='beginner_higher_energy')
rule2_intensity = ctrl.Rule(
                  antecedent=(physical_fitness['beginner'] & (energy['medium'] | energy['high'])),
                  consequent=(exercise_intensity['low'] | exercise_intensity['medium']),
                  label='beginner_higher_energy_intensity'
)

rule2_time = ctrl.Rule(
                  antecedent=(physical_fitness['beginner'] & (energy['medium'] | energy['high'])),
                  consequent=exercise_time['medium'],
                  label='beginner_higher_energy_time'
)

# rule3 = ctrl.Rule(antecedent=(physical_fitness['intermediate'] & energy['high']),
#                   consequent=(exercise_intensity['medium'] | exercise_intensity['high'],
#                               exercise_time['medium'] | exercise_time['long']),
#                   label='intermediate_high_energy')
rule3_intensity = ctrl.Rule(
                  antecedent=(physical_fitness['intermediate'] & energy['high']),
                  consequent=(exercise_intensity['medium'] | exercise_intensity['high']),
                  label='intermediate_high_energy_intensity')

rule3_time = ctrl.Rule(
                  antecedent=(physical_fitness['intermediate'] & energy['high']),
                  consequent=(exercise_time['medium'] | exercise_time['long']),
                  label='intermediate_high_energy_time')

# rule4 = ctrl.Rule(antecedent=(sports_goal['weight loss'] & age['elderly']),
#                   consequent=(exercise_intensity['low'] | exercise_intensity['medium'],
#                               exercise_time['short']),
#                   label='weight_loss_elderly')
rule4_intensity = ctrl.Rule(
                  antecedent=(sports_goal['weight loss'] & age['elderly']),
                  consequent=(exercise_intensity['low'] | exercise_intensity['medium']),
                  label='weight_loss_elderly_intensity')

rule4_time = ctrl.Rule(
                  antecedent=(sports_goal['weight loss'] & age['elderly']),
                  consequent=(exercise_time['short']),
                  label='weight_loss_elderly_time')

# rule5 = ctrl.Rule(antecedent=(sports_goal['muscle gain'] & age['young']),
#                   consequent=(exercise_intensity['high'],
#                               exercise_time['medium'] | exercise_time['long']),
#                   label='muscle_gain_young')

rule5_intensity = ctrl.Rule(
                  antecedent=(sports_goal['muscle gain'] & age['young']),
                  consequent=(exercise_intensity['high']),
                  label='muscle_gain_young_intensity')

rule5_time = ctrl.Rule(antecedent=(sports_goal['muscle gain'] & age['young']),
                  consequent=(exercise_time['medium'] | exercise_time['long']),
                  label='muscle_gain_young_time')

rule6 = ctrl.Rule(
                  antecedent=(sports_goal['general fitness'] & physical_fitness['intermediate'] & energy['medium']),
                  consequent=(exercise_intensity['medium'],exercise_time['medium']),
                  label='general_fitness')

# rule7 = ctrl.Rule(antecedent=(weight['overweight'] & sports_goal['weight loss']),
#                   consequent=(exercise_intensity['low'] | exercise_intensity['medium'],
#                               exercise_time['short']),
#                   label='overweight_weight_loss')
rule7_intensity = ctrl.Rule(
                  antecedent=(weight['overweight'] & sports_goal['weight loss']),
                  consequent=(exercise_intensity['low'] | exercise_intensity['medium']),
                  label='overweight_weight_loss_intensity')

rule7_time = ctrl.Rule(
                  antecedent=(weight['overweight'] & sports_goal['weight loss']),
                  consequent=(exercise_time['short']),
                  label='overweight_weight_loss_time')

rule8 = ctrl.Rule(
                       antecedent=(physical_fitness['intermediate'] & energy['low']),
                       consequent=(exercise_intensity['low'], exercise_time['short']),
                       label='intermediate_low_energy')

rule9 = ctrl.Rule(
                       antecedent=(physical_fitness['advanced'] & energy['high']),
                       consequent=(exercise_intensity['high'], exercise_time['long']),
                       label='advanced_high_energy')

rule10 = ctrl.Rule(
                       antecedent=(age['young'] & sports_goal['general fitness']),
                       consequent=(exercise_intensity['medium'], exercise_time['medium']),
                       label='young_general_fitness')

# rule11 = ctrl.Rule(antecedent=(sports_goal['muscle gain'] & age['middle-aged']),
#                        consequent=(exercise_intensity['medium'] | exercise_intensity['high'], exercise_time['short']),
#                        label='middle_aged_muscle_gain')
rule11_intensity = ctrl.Rule(
                       antecedent=(sports_goal['muscle gain'] & age['middle-aged']),
                       consequent=(exercise_intensity['medium'] | exercise_intensity['high']),
                       label='middle_aged_muscle_gain_intensity')

rule11_time = ctrl.Rule(
                       antecedent=(sports_goal['muscle gain'] & age['middle-aged']),
                       consequent=(exercise_time['short']),
                       label='middle_aged_muscle_gain_time')

# rule12 = ctrl.Rule(antecedent=(weight['overweight'] & energy['high'] & sports_goal['general fitness']),
#                        consequent=(exercise_intensity['medium'] | exercise_intensity['high'], exercise_time['medium']),
#                        label='overweight_high_energy_general_fitness')
rule12_intensity = ctrl.Rule(
                       antecedent=(weight['overweight'] & energy['high'] & sports_goal['general fitness']),
                       consequent=(exercise_intensity['medium'] | exercise_intensity['high']),
                       label='overweight_high_energy_general_fitness_intensity')

rule12_time = ctrl.Rule(
                       antecedent=(weight['overweight'] & energy['high'] & sports_goal['general fitness']),
                       consequent=(exercise_time['medium']),
                       label='overweight_high_energy_general_fitness_time')





# Fuzzy Control System:
all_rules = [rule1, rule2_intensity, rule2_time, rule3_intensity, rule3_time, rule4_intensity, rule4_time,
             rule5_intensity, rule5_time, rule6, rule7_intensity, rule7_time, rule8, rule9, rule10,
             rule11_intensity, rule11_time,rule12_intensity, rule12_time]


exercise_ctrl = ctrl.ControlSystem(all_rules)
exercise_simulation = ctrl.ControlSystemSimulation(exercise_ctrl)


# samples:
exercise_simulation.input['physical_fitness'] = 7
exercise_simulation.input['energy'] = 8
exercise_simulation.input['sports_goal'] = 6
exercise_simulation.input['age'] = 30
exercise_simulation.input['weight'] = 75

exercise_simulation.compute()

print("Intensity:", exercise_simulation.output['exercise_intensity'])
print("Time:", exercise_simulation.output['exercise_time'])