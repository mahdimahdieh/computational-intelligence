import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


Soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'Soil_moisture')
Weather_condition = ctrl.Antecedent(np.arange(0, 101, 1), 'Weather_condition')
Irrigation_amount = ctrl.Consequent(np.arange(0, 101, 1), 'Irrigation_amount')

Soil_moisture['dry'] = fuzz.trapmf(Soil_moisture.universe, [0, 0, 20, 40])
Soil_moisture['medium'] = fuzz.gaussmf(Soil_moisture.universe, 50, 12)
Soil_moisture['wet'] = fuzz.trapmf(Soil_moisture.universe, [60, 80, 100, 100])

Weather_condition['sunny'] = fuzz.trapmf(Weather_condition.universe, [0, 0, 30, 50])
Weather_condition['cloudy'] = fuzz.gaussmf(Weather_condition.universe, 60, 10)
Weather_condition["rainy"] = fuzz.trapmf(Weather_condition.universe, [50, 70, 100, 100])

Irrigation_amount['none'] = fuzz.trimf(Irrigation_amount.universe, [0, 0, 15])
Irrigation_amount['low'] = fuzz.trimf(Irrigation_amount.universe, [10, 25, 40])
Irrigation_amount['medium'] = fuzz.trimf(Irrigation_amount.universe, [35, 50, 65])
Irrigation_amount['high'] = fuzz.trapmf(Irrigation_amount.universe, [60, 80, 100, 100])



# Related to soil moisture level:
plt.figure(figsize=(8, 4))
plt.plot(Soil_moisture.universe, Soil_moisture['dry'].mf, 'b', label='Dry', linewidth=1.5)
plt.plot(Soil_moisture.universe, Soil_moisture['medium'].mf, 'g', label='Medium', linewidth=1.5)
plt.plot(Soil_moisture.universe, Soil_moisture['wet'].mf, 'r', label='Wet', linewidth=1.5)
plt.title('Soil Moisture Membership Functions')
plt.xlabel('Moisture Level (%)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()

# Related to weather conditions:
plt.figure(figsize=(8, 4))
plt.plot(Weather_condition.universe, Weather_condition['sunny'].mf, 'y', label='Sunny', linewidth=1.5)
plt.plot(Weather_condition.universe, Weather_condition['cloudy'].mf, 'gray', label='Cloudy', linewidth=1.5)
plt.plot(Weather_condition.universe, Weather_condition['rainy'].mf, 'b', label='Rainy', linewidth=1.5)
plt.title('Weather Condition Membership Functions')
plt.xlabel('Weather Condition (%)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()

# Related to the amount of irrigation:
plt.figure(figsize=(8, 4))
plt.plot(Irrigation_amount.universe, Irrigation_amount['none'].mf, 'gray', label='None', linewidth=1.5)
plt.plot(Irrigation_amount.universe, Irrigation_amount['low'].mf, 'b', label='Low', linewidth=1.5)
plt.plot(Irrigation_amount.universe, Irrigation_amount['medium'].mf, 'g', label='Medium', linewidth=1.5)
plt.plot(Irrigation_amount.universe, Irrigation_amount['high'].mf, 'r', label='High', linewidth=1.5)
plt.title('Irrigation Amount Membership Functions')
plt.xlabel('Irrigation Amount (%)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()



rule1 = ctrl.Rule(Soil_moisture['dry'] & Weather_condition['sunny'], Irrigation_amount['high'])
rule2 = ctrl.Rule(Soil_moisture['dry'] & Weather_condition['cloudy'], Irrigation_amount['medium'])
rule3 = ctrl.Rule(Soil_moisture['dry'] & Weather_condition['rainy'], Irrigation_amount['low'])
rule4 = ctrl.Rule(Soil_moisture['medium'] & Weather_condition['sunny'], Irrigation_amount['medium'])
rule5 = ctrl.Rule(Soil_moisture['medium'] & Weather_condition['cloudy'], Irrigation_amount['low'])
rule6 = ctrl.Rule(Soil_moisture['medium'] & Weather_condition['rainy'], Irrigation_amount['none'])
rule7 = ctrl.Rule(Soil_moisture['wet'] & Weather_condition['sunny'], Irrigation_amount['low'])
rule8 = ctrl.Rule(Soil_moisture['wet'] & Weather_condition['cloudy'], Irrigation_amount['none'])
rule9 = ctrl.Rule(Soil_moisture['wet'] & Weather_condition['rainy'], Irrigation_amount['none'])

# Fuzzy Control System:
irrigation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
irrigation_system = ctrl.ControlSystemSimulation(irrigation_ctrl)



# مقادیر نمونه ورودی (فرضی)
irrigation_system.input['Soil_moisture'] = 30  # رطوبت خاک 30%
irrigation_system.input['Weather_condition'] = 40  # شرایط آب‌وهوا 40% (بین آفتابی و ابری)

# محاسبه خروجی فازی
# irrigation_system.compute()



# Defuzzification:

centroid = fuzz.defuzz(Irrigation_amount.universe, Irrigation_amount.output,'centroid')
print(f"Centroid: {centroid:.2f}%")

mom = fuzz.defuzz(Irrigation_amount.universe, Irrigation_amount.output, 'mom')
print(f"Mean of Max (MOM): {mom:.2f}%")

lom = fuzz.defuzz(Irrigation_amount.universe, Irrigation_amount.output, 'lom')
print(f"Largest of Max (LOM): {lom:.2f}%")

som = fuzz.defuzz(Irrigation_amount.universe, Irrigation_amount.output, 'som')
print(f"Smallest of Max (SOM): {som:.2f}%")

bisector = fuzz.defuzz(Irrigation_amount.universe, Irrigation_amount.output, 'bisector')
print(f"Bisector: {bisector:.2f}%")