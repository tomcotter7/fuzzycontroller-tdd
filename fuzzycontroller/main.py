from fuzzycontroller.system.singleton import SingletonFIS
from fuzzycontroller.system.nonsingleton import NonSingletonFIS
import os

# Singleton FIS

print("Singleton FIS")
print("------------")
sfis = SingletonFIS()
sfis.load_data('./fuzzycontroller/example.json')
temp = float(input("Enter temperature: "))
headache = float(input("Enter headache: "))
age = float(input("Enter age: "))
inputs = {"temperature": temp, "headache": headache, "age": age}
output = sfis.compute_defuzzified_output(inputs)
print("Output: ", output)
sfis.graph_membership_functions()

# Non-Singleton FIS

print("Non-Singleton FIS")
print("-----------------")
nsfis = NonSingletonFIS()
nsfis.load_data('./fuzzycontroller/example.json')
temp_start = float(input("Enter temperature start: "))
temp_end = float(input("Enter temperature end: "))
headache_start = float(input("Enter headache start: "))
headache_end = float(input("Enter headache end: "))
age_start = float(input("Enter age start: "))
age_end = float(input("Enter age end: "))
inputs = {"temperature": {"start": temp_start, "end": temp_end},
          "headache": {"start": headache_start, "end": headache_end},
          "age": {"start": age_start, "end": age_end}}
output = nsfis.compute_defuzzified_output(inputs)
print("Output: ", output)
nsfis.graph_membership_functions()
