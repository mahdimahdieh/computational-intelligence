import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


x_soil = np.arange(0, 101, 1)
x_weather = np.arange(0, 101, 1)
x_irrigation = np.arange(0, 101, 1)

# Creating membership functions:

soil_dry = fuzz.trapmf(x_soil, [0, 0, 20, 40])
soil_medium = fuzz.gaussmf(x_soil, 50, 12)
soil_wet = fuzz.trapmf(x_soil, [60, 80, 100, 100])

weather_sunny = fuzz.trapmf(x_weather, [0, 0, 30, 50])
weather_cloudy = fuzz.gaussmf(x_weather, 60, 10)
weather_rainy = fuzz.trapmf(x_weather, [50, 70, 100, 100])

irrigation_none = fuzz.trimf(x_irrigation, [0, 0, 15])
irrigation_low = fuzz.trimf(x_irrigation, [10, 25, 40])
irrigation_medium = fuzz.trimf(x_irrigation, [35, 50, 65])
irrigation_high = fuzz.trapmf(x_irrigation, [60, 80, 100, 100])

# Show membership functions:

## For Soil Moisture Level:
plt.figure(figsize=(8, 4))
plt.plot(x_soil, soil_dry, 'b', label='Dry', linewidth=1.5)
plt.plot(x_soil, soil_medium, 'g', label='Medium', linewidth=1.5)
plt.plot(x_soil, soil_wet, 'r', label='Wet', linewidth=1.5)
plt.title('Soil Moisture Membership Functions')
plt.xlabel('Moisture Level (%)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()

## For Weather Condition:
plt.figure(figsize=(8, 4))
plt.plot(x_weather, weather_sunny, 'y', label='Sunny', linewidth=1.5)
plt.plot(x_weather, weather_cloudy, 'gray', label='Cloudy', linewidth=1.5)
plt.plot(x_weather, weather_rainy, 'b', label='Rainy', linewidth=1.5)
plt.title('Weather Condition Membership Functions')
plt.xlabel('Weather Condition (%)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()

## For Irrigation Amount:
plt.figure(figsize=(8, 4))
plt.plot(x_irrigation, irrigation_none, 'gray', label='None', linewidth=1.5)
plt.plot(x_irrigation, irrigation_low, 'b', label='Low', linewidth=1.5)
plt.plot(x_irrigation, irrigation_medium, 'g', label='Medium', linewidth=1.5)
plt.plot(x_irrigation, irrigation_high, 'r', label='High', linewidth=1.5)
plt.title('Irrigation Amount Membership Functions')
plt.xlabel('Irrigation Amount (%)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()

# Fuzzy Inference:

# input sample:
input_soil = 30
input_weather = 40

# Calculate the membership degrees of the inputs:
mu_soil_dry = fuzz.interp_membership(x_soil, soil_dry, input_soil)
mu_soil_medium = fuzz.interp_membership(x_soil, soil_medium, input_soil)
mu_soil_wet = fuzz.interp_membership(x_soil, soil_wet, input_soil)

mu_weather_sunny = fuzz.interp_membership(x_weather, weather_sunny, input_weather)
mu_weather_cloudy = fuzz.interp_membership(x_weather, weather_cloudy, input_weather)
mu_weather_rainy = fuzz.interp_membership(x_weather, weather_rainy, input_weather)

#Rules:

#if soil is dry AND weather is sunny then irrigation is high:
rule1_activation = np.fmin(mu_soil_dry, mu_weather_sunny)
#if soil is dry AND weather is cloudy then irrigation is medium:
rule2_activation = np.fmin(mu_soil_dry, mu_weather_cloudy)
#if soil is dry AND weather is rainy then irrigation is low:
rule3_activation = np.fmin(mu_soil_dry, mu_weather_rainy)
#if soil is medium AND weather is sunny then irrigation is medium:
rule4_activation = np.fmin(mu_soil_medium, mu_weather_sunny)
#if soil is medium AND weather is cloudy then irrigation is low:
rule5_activation = np.fmin(mu_soil_medium, mu_weather_cloudy)
#if soil is medium AND weather is rainy then irrigation is none:
rule6_activation = np.fmin(mu_soil_medium, mu_weather_rainy)
#if soil is wet AND weather is sunny then irrigation is low:
rule7_activation = np.fmin(mu_soil_wet, mu_weather_sunny)
#if soil is wet AND weather is cloudy then irrigation is none:
rule8_activation = np.fmin(mu_soil_wet, mu_weather_cloudy)
#if soil is wet AND weather is rainy then irrigation is none:
rule9_activation = np.fmin(mu_soil_wet, mu_weather_rainy)

# Maximizing operations:
activation_high = rule1_activation
activation_medium = np.fmax(rule2_activation, rule4_activation)
activation_low = np.fmax(rule3_activation, np.fmax(rule5_activation, rule7_activation))
activation_none = np.fmax(rule6_activation, np.fmax(rule8_activation, rule9_activation))


irrigation_none_clip = np.fmin(activation_none, irrigation_none)
irrigation_low_clip = np.fmin(activation_low, irrigation_low)
irrigation_medium_clip = np.fmin(activation_medium, irrigation_medium)
irrigation_high_clip = np.fmin(activation_high, irrigation_high)


aggregated_mf = np.fmax(irrigation_none_clip, np.fmax(irrigation_low_clip, np.fmax(irrigation_medium_clip, irrigation_high_clip)))



plt.figure(figsize=(8, 4))
plt.plot(x_irrigation, aggregated_mf, 'm', label='Aggregated MF', linewidth=2)
plt.title('Aggregated Output Membership Function')
plt.xlabel('Irrigation Amount (%)')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid()
plt.show()

# Defuzzification:
centroid_val = fuzz.defuzz(x_irrigation, aggregated_mf, 'centroid')
mom_val      = fuzz.defuzz(x_irrigation, aggregated_mf, 'mom')
lom_val      = fuzz.defuzz(x_irrigation, aggregated_mf, 'lom')
som_val      = fuzz.defuzz(x_irrigation, aggregated_mf, 'som')
bisector_val = fuzz.defuzz(x_irrigation, aggregated_mf, 'bisector')

print(f"Centroid: {centroid_val:.2f}%")
print(f"Mean of Maximum (MOM): {mom_val:.2f}%")
print(f"Largest of Maximum (LOM): {lom_val:.2f}%")
print(f"Smallest of Maximum (SOM): {som_val:.2f}%")
print(f"Bisector: {bisector_val:.2f}%")
