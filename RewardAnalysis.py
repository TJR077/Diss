#!/usr/bin/env python
# coding: utf-8

#import SUMO comands and custom environment
import traci
import numpy as np
from SUMOEnvironmentDevelopment import SUMOEnv
from SUMOEnvironmentDevelopment import PolicyManager



data1=[]
data2=[]

#set up for number of iterations
for i in range(50):
    env=SUMOEnv("testsim1.sumocfg") #import scenario
    LV=[]
    RV=[]
    #set up for number of episodes of each iteration
    for i in range(100):
        #Open SUMO and reset values
        env.reset()
        #Run simulation until no vehicles left
        while traci.simulation.getMinExpectedNumber() > 0:
            env.step()
        
        #apply rewards and store them
        env.apply_final_rewards(env.lane_change_time,env.Time_to_main)
        LV.append(env.rewards["lane_changer"])
        RV.append(env.rewards["give_way"])
        #close SUMO
        traci.close()
    
    data1.append(LV)
    data2.append(RV)

#save rewards given every episode
np.save('testsim1_LV.npy', data1)
np.save('testsim1_RV.npy', data2)