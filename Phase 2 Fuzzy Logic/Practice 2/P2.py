import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# *****************************
# Part 1: Define membership functions
# *****************************

# Domains:
x_soil = np.arange(0, 101, 1)        # 0-100 %
x_weather = np.arange(0, 101, 1)     # 0-100 %
x_irrigation = np.arange(0, 11, 1)  # 0-10 min or litter

# Soil moisture membership functions:
soil_dry    = fuzz.trapmf(x_soil, [0, 0, 20, 40])
soil_medium = fuzz.trimf(x_soil, [30, 50, 70])
soil_wet    = fuzz.trapmf(x_soil, [60, 80, 100, 100])

# Weather condition membership functions:
weather_sunny  = fuzz.trapmf(x_weather, [0, 0, 10, 25])
weather_cloudy = fuzz.trimf(x_weather, [20, 50, 80])
weather_rainy  = fuzz.trapmf(x_weather, [60, 85, 100, 100])

# Irrigation amount membership functions:
irrigation_none   = fuzz.trapmf(x_irrigation, [0, 0, 1, 2])
irrigation_low    = fuzz.trimf(x_irrigation, [1, 3, 4])
irrigation_medium = fuzz.trimf(x_irrigation, [3, 5, 7])
irrigation_high   = fuzz.trapmf(x_irrigation, [6, 8, 10, 10])

# *****************************
# Plot membership functions
# *****************************

# for Soil Moisture:
plt.figure(figsize=(8, 5))
plt.plot(x_soil, soil_dry, label='Dry')
plt.plot(x_soil, soil_medium, label='Medium')
plt.plot(x_soil, soil_wet, label='Wet')
plt.title('Soil Moisture Membership Functions')
plt.xlabel('Soil Moisture (%)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()

# for Weather Condition:
plt.figure(figsize=(8, 5))
plt.plot(x_weather, weather_sunny, label='Sunny')
plt.plot(x_weather, weather_cloudy, label='Cloudy')
plt.plot(x_weather, weather_rainy, label='Rainy')
plt.title('Weather Condition Membership Functions')
plt.xlabel('Weather Condition (%)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()

# for Irrigation Amount:
plt.figure(figsize=(8, 5))
plt.plot(x_irrigation, irrigation_none, label='None')
plt.plot(x_irrigation, irrigation_low, label='Low')
plt.plot(x_irrigation, irrigation_medium, label='Medium')
plt.plot(x_irrigation, irrigation_high, label='High')
plt.title('Irrigation Amount Membership Functions')
plt.xlabel('Irrigation Amount (min or Lit)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()


def fuzzify_inputs(soil_val, weather_val):
    mu_soil = {
        'dry':    fuzz.interp_membership(x_soil, soil_dry, soil_val),
        'medium': fuzz.interp_membership(x_soil, soil_medium, soil_val),
        'wet':    fuzz.interp_membership(x_soil, soil_wet, soil_val)
    }
    mu_weather = {
        'sunny':  fuzz.interp_membership(x_weather, weather_sunny, weather_val),
        'cloudy': fuzz.interp_membership(x_weather, weather_cloudy, weather_val),
        'rainy':  fuzz.interp_membership(x_weather, weather_rainy, weather_val)
    }
    return mu_soil, mu_weather


def infer_irrigation(mu_soil, mu_weather):
    # Rules:


    # if soil is dry AND weather is sunny then irrigation is high:
    rule1 = np.fmin(mu_soil['dry'],    mu_weather['sunny'])
    # if soil is dry AND weather is cloudy then irrigation is medium:
    rule2 = np.fmin(mu_soil['dry'],    mu_weather['cloudy'])
    # if soil is dry AND weather is rainy then irrigation is low:
    rule3 = np.fmin(mu_soil['dry'],    mu_weather['rainy'])
    # if soil is medium AND weather is sunny then irrigation is medium:
    rule4 = np.fmin(mu_soil['medium'], mu_weather['sunny'])
    # if soil is medium AND weather is cloudy then irrigation is low:
    rule5 = np.fmin(mu_soil['medium'], mu_weather['cloudy'])
    # if soil is medium AND weather is rainy then irrigation is none:
    rule6 = np.fmin(mu_soil['medium'], mu_weather['rainy'])
    # if soil is wet AND weather is sunny then irrigation is low:
    rule7 = np.fmin(mu_soil['wet'],    mu_weather['sunny'])
    # if soil is wet AND weather is cloudy then irrigation is none:
    rule8 = np.fmin(mu_soil['wet'],    mu_weather['cloudy'])
    # if soil is wet AND weather is rainy then irrigation is none:
    rule9 = np.fmin(mu_soil['wet'],    mu_weather['rainy'])

    # Aggregation of rule outputs
    high_activation   = rule1 #Only rule 1 produces high irrigation.
    medium_activation = np.fmax(rule2, rule4)
    low_activation    = np.fmax(rule3, np.fmax(rule5, rule7))
    none_activation   = np.fmax(rule6, np.fmax(rule8, rule9))

    # Clip output MFs
    high_clip   = np.fmin(high_activation,   irrigation_high)
    medium_clip = np.fmin(medium_activation, irrigation_medium)
    low_clip    = np.fmin(low_activation,    irrigation_low)
    none_clip   = np.fmin(none_activation,   irrigation_none)

    # Aggregate all clipped output MFs
    aggregated = np.fmax(none_clip,
                  np.fmax(low_clip,
                  np.fmax(medium_clip, high_clip)))
    return aggregated


def defuzzify_output(aggregated_mf, method='centroid'):
    if method in ['centroid', 'mom', 'lom', 'som', 'bisector']:
        return fuzz.defuzz(x_irrigation, aggregated_mf, method)
    else:
        raise ValueError(f"Unknown defuzz method: {method}")

# *****************************
# Part 2: Example and defuzzification comparison
# *****************************
input_soil = 30
input_weather = 40

mu_s, mu_w = fuzzify_inputs(input_soil, input_weather)
agg = infer_irrigation(mu_s, mu_w)

methods = ['centroid', 'mom', 'lom', 'som', 'bisector']
print("Defuzzification results for sample input:")
for m in methods:
    print(f"{m}: {defuzzify_output(agg, m):.2f}")



# *****************************
# Part 3: Simulation over time
# *****************************

# 1. Define initial conditions and weather sequence for 10 days
initial_soil = 15  # meghdar avallieh baraye rotubat khaak
weather_sequence = ['sunny', 'sunny', 'cloudy', 'rainy', 'sunny',
                    'cloudy', 'rainy', 'sunny', 'cloudy', 'rainy']

# Mapping from weather label to numeric
weather_val_map   = {'sunny': 0,   'cloudy': 50,  'rainy': 100}
effect_map        = {'sunny': -5,  'cloudy': -2,  'rainy': +5}

# Histories simulation
soil_history       = [initial_soil]
irrigation_history = []
days               = list(range(1, len(weather_sequence) + 1))

# Run simulation
print("\nSimulating irrigation in 10 Days:")
for day, weather_label in zip(days, weather_sequence):
    current_soil = soil_history[-1]
    w_val = weather_val_map[weather_label]

    # Fuzzify inputs
    mu_s, mu_w = fuzzify_inputs(current_soil, w_val)
    agg   = infer_irrigation(mu_s, mu_w)
    irr   = defuzzify_output(agg, method='centroid')

    # Record irrigation
    irrigation_history.append(irr)

    # Update soil moisture: add weather effect and irrigation contribution
    weather_effect = effect_map[weather_label]
    new_soil = current_soil + (0.6 * irr) - weather_effect
    new_soil = np.clip(new_soil, 0, 100)
    soil_history.append(new_soil)

    print(f"Day {day}: Weather={weather_label}, Soil before={current_soil:.1f}%, "
          f"Irrigation={irr:.2f}, Soil after={new_soil:.1f}%")

# Plot results
plt.figure(figsize=(8, 4))
plt.plot([0] + days, soil_history, marker='o')
plt.title('Soil Moisture Over Time')
plt.xlabel('Day')
plt.ylabel('Soil Moisture (%)')
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(days, irrigation_history)
plt.title('Daily Irrigation Amount')
plt.xlabel('Day')
plt.ylabel('Irrigation (min or L)')
plt.grid(axis='y')
plt.show()

soil_values = soil_history[1:]
plt.figure(figsize=(8, 4))
plt.bar(days, soil_values)
plt.title('Daily Soil Moisture')
plt.xlabel('Day')
plt.ylabel('Soil Moisture (%)')
plt.grid(axis='y')
plt.show()