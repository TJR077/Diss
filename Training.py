# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:06:36 2024

@author: thoma
"""
#import SUMO commands and custom environment
import traci
import numpy as np
from SUMOEnvironmentDevelopment import SUMOEnv
from SUMOEnvironmentDevelopment import PolicyManager

#set up list of scenarios
sumo=[]
for i in range(1,37):
    sumo.append(f"train{i}.sumocfg")

#initiate environment
env=SUMOEnv(sumo)

#set up number of simulations for Training the Policy
for i in range(20000):
    #reset variables and open SUMO
    env.reset()
    #run simulation
    while traci.simulation.getMinExpectedNumber() > 0:
        env.step()
    #apply rewards and close SUMO
    env.apply_final_rewards(env.lane_change_time,env.Time_to_main)
    traci.close()

#aquire trained policy and save as npy files
policy1=env.policy_managers["lane_changer"].policies
policy2=env.policy_managers["give_way"].policies
np.save('lane_change_policy.npy', policy1)
np.save('give_way_policy.npy', policy2)


        