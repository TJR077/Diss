# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:11:55 2024

@author: thoma
"""

import traci
import numpy as np




class SUMOEnv():
    
    def __init__(self,sumo,lane_change_policy,give_way_policy):
        self.agents = ["lane_changer","give_way"] #defining possible agents
        self.possible_agents = self.agents[:]
        self.sumo_configs = [sumo]
        self.action_spaces = {"lane_changer": [0,1],  # 0: stay in lane, 1: change lane
                                "give_way": [0,1]  # 0: don't give way, 1: give way
                                }


        # parameters for bins within the policy matrix
        self.num_bins_1 = 20
        self.num_bins_2 = 20
        self.min_var_value_1 = {"lane_changer": 0,"give_way": 0}  
        self.max_var_value_1 = {"lane_changer": 784,"give_way": 14} 
        self.min_var_value_2 = {"lane_changer": 0,"give_way": -56} 
        self.max_var_value_2 = {"lane_changer": 200,"give_way": 56}
        
        #input external trained policy
        self.LV_policy = lane_change_policy
        self.RV_policy = give_way_policy
        
    def identify_agents(self):
        #function to recognise what each agents id is
        LV = False
        RV = False
        FV = False
        vehicle_ids = traci.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            lane_index = traci.vehicle.getLaneID(vehicle_id)
            if lane_index == '1to1b_0': #only works for this junction
                LV = vehicle_id
                FV_try = traci.vehicle.getLeftLeaders(vehicle_id)
                RV_try = traci.vehicle.getLeftFollowers(vehicle_id)
                if len(FV_try)>0:       #if no front vehicle keep as false
                    FV = FV_try[0][0]
                
                if len(RV_try)>0:       # if no rear vehicle keep as false
                    RV = RV_try[0][0]
        
        return {"lane_changer":LV,"give_way":RV,"front_vehicle":FV}
            
    
    def get_variables(self,vehicle_id,other_vehicle_id,front_vehicle_id):
        #defines variables for each vehicle
        #creates dummy variables for False IDs only relevant for this junction
        traffic_light_time = traci.trafficlight.getNextSwitch('main')        
        if vehicle_id==False:   #incase RV is false 
            speed=10            #has to be non zero value
            lane_pos=0
            lane_index=1
            distance_to_light=400
            max_speed = 13.9
        else:
            #collect all useful values
            speed = traci.vehicle.getSpeed(vehicle_id)
            lane_pos = traci.vehicle.getDistance(vehicle_id)
            lane_index = traci.vehicle.getLaneIndex(vehicle_id)
            max_speed = traci.vehicle.getAllowedSpeed(vehicle_id)
            distance_to_light = traci.vehicle.getDrivingDistance(vehicle_id,'1btomain',5)
            if distance_to_light<0:
                distance_to_light=0

        if other_vehicle_id==False: #incase other_vehicle_id is a False RV
            other_speed = 10
            other_lane_pos = 0
        else:
            other_speed = traci.vehicle.getSpeed(other_vehicle_id)
            other_lane_pos = traci.vehicle.getDistance(other_vehicle_id)
            
        if front_vehicle_id==False:
            front_lane_pos= 800
        else:
            front_lane_pos = traci.vehicle.getDistance(front_vehicle_id)
            
        relative_speed = speed - other_speed
        relative_position = lane_pos - other_lane_pos
        row = front_lane_pos - max(lane_pos,other_lane_pos)
        
        
        return np.array([speed,lane_pos,traffic_light_time,distance_to_light,relative_speed,
                        relative_position,row,lane_index,max_speed])
    
    def get_actions(self, variables, LV_id):
        #choose actions based on the current variables using the policy managers
        chosen_actions = {}
        for agent in self.agents:
            full_var = variables[agent]
            max_game_distance = full_var[8]*4 #4seconds x max speed
            if agent == "lane_changer":
                var_1 = min(full_var[5],max_game_distance)*min(full_var[6],max_game_distance) #custom variable 1 for LV
                var_2 = full_var[3] #custom variable 2 for LV
        
                if var_1>1000: #if LV has this much space should change lanes
                    chosen_actions[agent]=1
                
                if LV_id in self.change_time: #commit to lane change once decided
                    chosen_actions[agent]=1
                        
                else: #choose action based off policy
                    chosen_actions[agent] = np.random.choice(
                            self.action_spaces[agent], 
                            p=self.get_policy_LV(var_1, var_2))
                    
            if agent == "give_way":
                var_1 = full_var[3]/full_var[2] #custom variable 1 for RV
                var_2 = full_var[5] #custom variable 2 for RV
                if var_1>full_var[8]: #means RV is not going to make lights so may as well give way
                    chosen_actions[agent] = 1
                if abs(var_2)>max_game_distance: #if the RV has that much space then doesn't have to consider LVs actions
                    chosen_actions[agent] = 0
                else: #choose actions based on policy
                    chosen_actions[agent] = np.random.choice(
                           self.action_spaces[agent], 
                           p=self.get_policy_RV(var_1, var_2))
                        
        return chosen_actions

    def give_way(self, rear_vehicle_id, lane_change_id, duration, factor):
        #collect all relavent variables
        rear_speed = traci.vehicle.getSpeed(rear_vehicle_id)
        lane_change_speed = traci.vehicle.getSpeed(lane_change_id)
        lane_change_lane_pos = traci.vehicle.getDistance(lane_change_id)
        rear_lane_pos = traci.vehicle.getDistance(rear_vehicle_id)
        gap = lane_change_lane_pos - rear_lane_pos

        #adjust the gap threshold and speed reduction parameters 
        gap_threshold = max(10, rear_speed * factor)  # ideal gap
        speed_reduction_factor = max(0.3, min(0.7, gap / gap_threshold))  #dynamic reduction factor
        #slowdown if gap less than ideal
        if gap < gap_threshold:
            reduced_speed = rear_speed * speed_reduction_factor
            traci.vehicle.slowDown(rear_vehicle_id, reduced_speed, duration)
        else: #match speed if gap is large enough
            traci.vehicle.slowDown(rear_vehicle_id, lane_change_speed, duration)
            

    def reset(self, seed=None, options=None):
        #choose random scenario from selection
        sumo_config = np.random.choice(self.sumo_configs)
        #open SUMO
        sumo_cmd = ["sumo", "-c", sumo_config]
        self.sumo_process = traci.start(sumo_cmd)
        #important to not use SUMO lane change model and have the vehicles start when defined
        for i in range(2):
            traci.vehicle.setLaneChangeMode(f"veh{i}", 0b000000000000)
        #reset all values
        self.steps = 0
        self.change_time = {}
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.variable_action_history = []
        self.lane_change_time = 0
        self.Time_to_main = 0
        self.collision = False
        self.change = False
        
        
    def reset_visual(self, seed=None, options=None):
        #choose random scenario from selection
        sumo_config = np.random.choice(self.sumo_configs)
        #open SUMO
        sumo_cmd = ["sumo-gui", "-c", sumo_config]
        self.sumo_process = traci.start(sumo_cmd)
        #important to not use SUMO lane change model and have the vehicles start when defined
        for i in range(2):
            traci.vehicle.setLaneChangeMode(f"veh{i}", 0b000000000000)
        #reset all values
        self.steps = 0
        self.change_time = {}
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.variable_action_history = []
        self.lane_change_time = 0
        self.Time_to_main = 0
        self.collision = False
        self.change = False


    def step(self):
        #function which runs simulation
        agent_ids=self.identify_agents()
        if agent_ids["lane_changer"]==False: #no actions if there is no LV
            traci.simulationStep()  # advance the simulation by one step
            self.steps += 1
            
        else:
            
            #get variables and actions of agents
            variable = {agent: self.get_variables(agent_ids[agent], agent_ids[self.agents[(idx+1)%2]], agent_ids["front_vehicle"]) for idx,
                     agent in enumerate(self.agents)}
        
            actions = self.get_actions(variable, agent_ids["lane_changer"])
            
            for agent in self.agents:
                if agent == "lane_changer":
                    vehicle_id = agent_ids["lane_changer"]
                    other_vehicle_id = agent_ids["give_way"]
                    lane_changer_action = actions["lane_changer"]
                    
                    if variable["lane_changer"][3]<100: #slow down as approaching junction to lane change
                        factor = max(variable["lane_changer"][3],0.5)
                        traci.vehicle.slowDown(vehicle_id, variable["lane_changer"][0]*factor, 0.1)
                        
                    if lane_changer_action == 1:
                        #set time for lane chage
                        if vehicle_id not in self.change_time:                    
                            delay_steps = 20  #number of steps to delay the lane change
                            self.change_time[vehicle_id] = self.steps + delay_steps  #schedule the lane change
                        
            
    
    
                elif agent == "give_way":
                    if agent_ids["give_way"]==False: #no action if theres no vehicle
                        pass
                    else:
                        vehicle_id = agent_ids["give_way"]
                        other_vehicle_id = agent_ids["lane_changer"]
                        give_way_action = actions["give_way"]
                        lane = traci.vehicle.getLaneID(vehicle_id)
                            
                        
                        if give_way_action == 1:  # give way
                            self.give_way(vehicle_id,other_vehicle_id,2,2)
        
            #implement lane change
            if len(self.change_time)>0:            
                for vehicle_id, scheduled_step in list(self.change_time.items()):
                    if self.steps >= scheduled_step:
                        x = traci.vehicle.getDistance(vehicle_id)
                        lane = '1to1b_1' #only functions for this junction 
                        traci.vehicle.moveTo(vehicle_id, lane, x) #lane change
                        self.change=True
                        if agent_ids["front_vehicle"]==False:
                            pass
                        else:
                            self.give_way(vehicle_id,agent_ids["front_vehicle"],2,1) #make sure it doesn't crash into FV
                        
                        self.lane_change_time = self.steps/10
                        if agent_ids["give_way"]==False:
                            self.Time_to_main=100
                        else:
                            self.Time_to_main = self.lane_change_time + variable["give_way"][3]/variable["give_way"][0]
                        
                        #visual representation of lane change
                        print(f"Lane change implemented for {vehicle_id} at step {self.steps}")
                    
                        #remove the vehicle from the schedule after lane change
                        del self.change_time[vehicle_id]
    
            traci.simulationStep()  #advance the simulation by one step
            colliding_vehicles = traci.simulation.getCollidingVehiclesIDList()
            #track collisions
            if len(colliding_vehicles)>0:
                self.collision=True
                print('collsion')
            self.steps += 1
            
    def get_policy_LV(self, var_1, var_2): #get policy out of selection matrix LV
        bin_width_1 = (self.max_var_value_1["lane_changer"]-self.min_var_value_1["lane_changer"])/self.num_bins_1
        bin_width_2 = (self.max_var_value_2["lane_changer"]-self.min_var_value_2["lane_changer"])/self.num_bins_2
        bin_index_1 = min(int((var_1-self.min_var_value_1["lane_changer"]) / bin_width_1), self.num_bins_1 - 1)
        bin_index_2 = min(int((var_2-self.min_var_value_2["lane_changer"]) / bin_width_2), self.num_bins_2 - 1)
        return self.LV_policy[bin_index_1, bin_index_2]

    def get_policy_RV(self, var_1, var_2): #get policy out of selection matrix LV
        bin_width_1 = (self.max_var_value_1["give_way"]-self.min_var_value_1["give_way"])/self.num_bins_1
        bin_width_2 = (self.max_var_value_2["give_way"]-self.min_var_value_2["give_way"])/self.num_bins_2
        bin_index_1 = min(int((var_1-self.min_var_value_1["give_way"]) / bin_width_1), self.num_bins_1 - 1)
        bin_index_2 = min(int((var_2-self.min_var_value_2["give_way"]) / bin_width_2), self.num_bins_2 - 1)
        return self.RV_policy[bin_index_1, bin_index_2]