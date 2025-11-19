import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Predicting risk of downy mildew with weather conditions

# Input Variables
temperature = ctrl.Antecedent(np.arange(-20, 120, 1), "temperature")
rainfall = ctrl.Antecedent(np.arange(0,305,1), "rainfall")
# Output
risk = ctrl.Consequent(np.arange(0,10,1), "risk")

# Temperature membership functions
temperature["low_risk"] = fuzz.zmf(temperature.universe, 7, 10)
temperature["high_risk"] = fuzz.smf(temperature.universe, 7, 10)
# temperature.view()
# plt.savefig("temp_risk.png")
# plt.clf()

rainfall["low_risk"] = fuzz.zmf(rainfall.universe, 7, 10)
rainfall["high_risk"] = fuzz.smf(rainfall.universe, 7, 10)
# rainfall.view()
# plt.savefig("rain_risk.png")
# plt.clf()

risk["low_risk"] = fuzz.trimf(risk.universe, [0,0,5])
risk["medium_risk"] = fuzz.trimf(risk.universe, [0,5,10])
risk["high_risk"] = fuzz.trimf(risk.universe, [5,10,10])
# risk.view()
# plt.savefig("risk.png")
# plt.clf()

# Rules
rule1 = ctrl.Rule(temperature["high_risk"] & rainfall["high_risk"], risk["high_risk"])
rule2 = ctrl.Rule(temperature["high_risk"] & rainfall["low_risk"], risk["medium_risk"])
rule3 = ctrl.Rule(temperature["low_risk"] & rainfall["high_risk"], risk["medium_risk"])
rule4 = ctrl.Rule(temperature["low_risk"] & rainfall["low_risk"], risk["low_risk"])

# Control mildew system
mildew_risk_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
mildew = ctrl.ControlSystemSimulation(mildew_risk_control)

mildew.input["temperature"] = 8
mildew.input["rainfall"] = 10

mildew.compute()

print(mildew.output["risk"])
risk.view(sim=mildew)
plt.savefig("mildew_output.png")
plt.clf()