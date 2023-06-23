import sys 
import os 
import numpy as np
import json
testing_file = sys.argv[1]

scenarios = []
actions = []
intents = []



with open(testing_file, "r") as tf : 
    lines = tf.read().splitlines()
    for line in lines  : 
        dic = json.loads(line)
        action_result = dic["action"] == dic["true_action"]
        scenario_result = dic["scenario"] == dic["true_scenario"] 
        intents.append(action_result and scenario_result)
        actions.append(action_result)
        scenarios.append(scenario_result)


print(f" scenario accuracy : {np.mean(scenarios)}")
print(f" action accuracy : {np.mean(actions)}")
print(f" intent accuracy : {np.mean(intents)}")

