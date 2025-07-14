# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:25:22 2024

@author: thoma
"""
#import SUMO control module and environment
import traci
import numpy as np
from SUMOTestEnvironment import SUMOEnv

#initiate lists for data
data1=[]
data2=[]
times = []

#load policies
LV_pol = np.load('lane_change_policy.npy', allow_pickle=True)
RV_pol = np.load('give_way_policy.npy', allow_pickle=True)

#initiate environment with scenario and policies
env=SUMOEnv("train11.sumocfg",LV_pol,RV_pol)


#for number of iterations wanted
for i in range(1):

    
    FVtoLV=np.zeros(50)
    LVtoRV=np.zeros(50)
    departure_times = {}
    #env.reset_visual()
    env.reset() #choose if visual or not
    post_change = 0
    #run simulation to completion
    while traci.simulation.getMinExpectedNumber() > 0:
        env.step()
        if env.change==True and post_change<50:
            #collect necessary data
            front_speed = traci.vehicle.getSpeed("veh0")
            front_pos = traci.vehicle.getDistance("veh0")
            front_lane = traci.vehicle.getLaneIndex("veh0")
    

            rear_speed = traci.vehicle.getSpeed("veh2")
            rear_pos = traci.vehicle.getDistance("veh2")
            rear_lane = traci.vehicle.getLaneIndex("veh2")

            LV_speed = traci.vehicle.getSpeed("veh1")
            LV_pos = traci.vehicle.getDistance("veh1")
            LV_lane = traci.vehicle.getLaneIndex("veh1")

            #calculate TTC between FV and LV avoiding divide by 0
            if (LV_speed-front_speed)==0:
                TTC2=30
            else:
                TTC2=(front_pos-LV_pos-4)/(LV_speed-front_speed)
                if TTC2>30 or TTC2<0:
                    TTC2=30
            FVtoLV[post_change]=TTC2
        
            #calculate TTC between LV and RV avoiding divide by 0
            if (rear_speed-LV_speed)==0:
                TTC3=30
            else:
                TTC3=(LV_pos-rear_pos-4)/(rear_speed-LV_speed)
                if TTC3>30 or TTC3<0:
                    TTC3=30
            LVtoRV[post_change]=TTC3
            
            post_change+=1
            
        #collect times for vehicles to complete simulation
        vehicles_left = traci.simulation.getArrivedIDList()
        current_time = traci.simulation.getTime()
    
        for vehicle_id in vehicles_left:
            departure_times[vehicle_id] = current_time        
            
    #collect data
    data1.append(FVtoLV)
    data2.append(LVtoRV)
    dep_times=[departure_times["veh0"],departure_times["veh1"],departure_times["veh2"]]
    times.append(dep_times)
    #close SUMO
    traci.close()
    


#save data
#np.save('FVtoLV8.npy', data1)
#np.save('LVtoRV8.npy', data2)
#np.save('times8.npy',times)

