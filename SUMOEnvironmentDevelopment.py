#!/usr/bin/env python
# coding: utf-8

import traci
import numpy as np

class SUMOEnv():
    
    def __init__(self,sumo):
        self.agents = ["lane_changer","give_way"] #defining possible agents
        self.sumo_configs = [sumo]
        self.action_spaces = {"lane_changer": [0,1],  # 0: stay in lane, 1: change lane
                                "give_way": [0,1]  # 0: don't give way, 1: give way
                                }

        #parameters for bins within the Policy Matrix
        self.num_bins_1 = 20
        self.num_bins_2 = 20
        self.min_var_value_1 = {"lane_changer": 0,"give_way": 0}  
        self.max_var_value_1 = {"lane_changer": 784,"give_way": 14} 
        self.min_var_value_2 = {"lane_changer": 0,"give_way": -56} 
        self.max_var_value_2 = {"lane_changer": 200,"give_way": 56}  

        #initiate policy managers for each agent
        self.policy_managers = {agent: PolicyManager(
                                    self.num_bins_1, 
                                    self.num_bins_2, 
                                    self.min_var_value_1[agent],
                                    self.max_var_value_1[agent],
                                    self.min_var_value_2[agent],
                                    self.max_var_value_2[agent]) 
                                for agent in self.agents}
        
        
    def identify_agents(self):
        #function to recognise what each agents id is
        LV = False
        RV = False
        FV = False
        vehicle_ids = traci.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            if lane_id == '1to1b_0': #lane id for target lane would have to be generalised for other junctions
                LV = vehicle_id
                FV_try = traci.vehicle.getLeftLeaders(vehicle_id)
                RV_try = traci.vehicle.getLeftFollowers(vehicle_id)
                if len(FV_try)>0:       #if no front vehicle keep as false
                    FV = FV_try[0][0]
                
                if len(RV_try)>0:       # if no rear vehicle keep as false
                    RV = RV_try[0][0]
        #return dictionary of IDs
        return {"lane_changer":LV,"give_way":RV,"front_vehicle":FV}
            
    
    def get_variables(self,vehicle_id,other_vehicle_id,front_vehicle_id):
        #defining variables for each vehicle
        #creates dummy variables for False IDs only relevant for this junction
        traffic_light_time = traci.trafficlight.getNextSwitch('main')        
        if vehicle_id==False:   #incase RV is false 
            speed=10            #has to be non zero value
            lane_pos=0
            lane_index=1
            distance_to_light=400
            max_speed = 13.9
        else:
            #get relevant information that could help later on
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
                
                if var_1>1000: #if value above this should just change lanes as has space helping to keep bins in range
                    chosen_actions[agent]=1
                
                #once vehicle decides to lane change it is commited for 2s before change
                if LV_id in self.change_time:
                    chosen_actions[agent]=1
                        
                else: #choose actions using policy
                    chosen_actions[agent] = np.random.choice(
                            self.action_spaces[agent], 
                            p=self.policy_managers[agent].get_policy(var_1, var_2))
                        
            if agent == "give_way":
                var_1 = full_var[3]/full_var[2] #custom variable 1 for RV
                var_2 = full_var[5] #custom variable to for RV
                if var_1>full_var[8]: #means that RV cannot make lights so should give way
                    chosen_actions[agent] = 1
                if abs(var_2)>max_game_distance: #if large enough gap does not need to consider LVs actions
                    chosen_actions[agent] = 0
                else:
                    #choose action on poliy
                    chosen_actions[agent] = np.random.choice(
                           self.action_spaces[agent], 
                           p=self.policy_managers[agent].get_policy(var_1, var_2))
                        
        return chosen_actions
                        
    
    def give_way(self, rear_vehicle_id, lane_change_id, duration, factor):
        rear_speed = traci.vehicle.getSpeed(rear_vehicle_id)
        lane_change_speed = traci.vehicle.getSpeed(lane_change_id)
        lane_change_lane_pos = traci.vehicle.getDistance(lane_change_id)
        rear_lane_pos = traci.vehicle.getDistance(rear_vehicle_id)
        gap = lane_change_lane_pos - rear_lane_pos

        #adjust the gap threshold and speed reduction parameters
        gap_threshold = max(10, rear_speed * factor)  #ideal gap to have
        speed_reduction_factor = max(0.3, min(0.7, gap / gap_threshold))  # dynamic reduction factor

        #slow down if gap below ideal gap
        if gap < gap_threshold:
            reduced_speed = rear_speed * speed_reduction_factor
            traci.vehicle.slowDown(rear_vehicle_id, reduced_speed, duration)
        else: #match speed if above ideal gap
            traci.vehicle.slowDown(rear_vehicle_id, lane_change_speed, duration)

    
    def calculate_rewards(self, lane_change_action, rear_vehicle_action, LV_var, RV_var, LC_time, RVtoTLtime):
        #beta calculation        
        tgl = RV_var[2]
        max_speed = RV_var[8]
        rear_speed = RV_var[0]
        rear_distance_to_light = RV_var[3]
        tgl_min = rear_distance_to_light/max_speed
        tgl_const = rear_distance_to_light/rear_speed
        if rear_vehicle_action == 0:
            beta = max(min((tgl_const-tgl_min)/(tgl - tgl_min),0.7),0.3)

        else:
            beta = max(min((tgl_const+5-tgl_min)/(tgl - tgl_min),0.7),0.3)
            
        #alpha calculation
        S_current = LV_var[1]
        S_change_max = 200
        S_change_min = 5
        alpha = max(min((S_change_max-S_current)/(S_change_max-S_change_min),0.7),0.3)
        
        #P safe calculation
        S_veh_min = 2.5
        S_veh_max = max_speed*2
        lane_change_lane_pos = LV_var[1]
        rear_lane_pos = RV_var[1]
        S_veh = min((lane_change_lane_pos - rear_lane_pos),80)
        P_safe = min((S_veh-S_veh_min)/(S_veh_max-S_veh_min),1)
    
        
        
        
        #space efficiency calculation
        L_space = LV_var[6]
        L_max = 80
        L_min = 2.5 
        P_s = min(L_space/(L_max-L_min),1)
        
        #RV efficiency calculation
        t_RV_min = tgl_min
        t_RV_max = 38
        t_RV_cur = RVtoTLtime 
        P_RV = P_s*min((t_RV_cur-t_RV_min)/(t_RV_max-t_RV_min),1)
        
        #LV efficiency calculation
        lane_change_speed = LV_var[0]
        lane_change_DTL=LV_var[3]
        t_LV_min = 0.1
        t_LV_max = lane_change_DTL/lane_change_speed
        t_LV_lc = LC_time
        P_LV = P_s*(min((t_LV_lc-t_LV_min)/(t_LV_max-t_LV_min),1))
        
        #apply -1 reward if there's a collision to offending actions
        if self.collision == True:
            if lane_change_action==0 and rear_vehicle_action ==0:
                return {"give_way":beta*P_RV+(1-beta)*P_safe,"lane_changer":-alpha*P_LV+(1-alpha)*P_safe}
            
            if lane_change_action==1 and rear_vehicle_action ==0:
                return {"give_way":-1,"lane_changer":-1}
            
            if lane_change_action==0 and rear_vehicle_action ==1:
                return {"give_way":-beta*P_RV+(1-beta)*P_safe,"lane_changer":-alpha*P_LV+(1-alpha)*P_safe}
            
            if lane_change_action==1 and rear_vehicle_action ==1:
                return {"give_way":-beta*P_RV+(1-beta)*P_safe,"lane_changer":-1}                    
        
        #otherwise apply rewards as defined
        else:
            if lane_change_action==0 and rear_vehicle_action ==0:
                return {"give_way":beta*P_RV+(1-beta)*P_safe,"lane_changer":-alpha*P_LV+(1-alpha)*P_safe}
            
            if lane_change_action==1 and rear_vehicle_action ==0:
                return {"give_way":-beta*P_RV-(1-beta)*P_safe,"lane_changer":-alpha*P_LV-(1-alpha)*P_safe}
            
            if lane_change_action==0 and rear_vehicle_action ==1:
                return {"give_way":-beta*P_RV+(1-beta)*P_safe,"lane_changer":-alpha*P_LV+(1-alpha)*P_safe}
            
            if lane_change_action==1 and rear_vehicle_action ==1:
                return {"give_way":-beta*P_RV+(1-beta)*P_safe,"lane_changer":alpha*P_LV+(1-alpha)*P_safe}
        
        

    def reset(self, seed=None, options=None):
        sumo_config = np.random.choice(self.sumo_configs) #randomly choose scenario from selection
        #open SUMO without visualisation
        sumo_cmd = ["sumo", "-c", sumo_config]
        self.sumo_process = traci.start(sumo_cmd)
        #important for vehicles not to use SUMO lane change model and start when defined
        for i in range(2):
            traci.vehicle.setLaneChangeMode(f"veh{i}", 0b000000000000)
        #reset all variables
        self.steps = 0      
        self.change_time = {}
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.variable_action_history = []
        self.lane_change_time = 0
        self.Time_to_main = 0
        self.collision = False
        
        
    def reset_visual(self, seed=None, options=None):
        sumo_config = np.random.choice(self.sumo_configs) #randomly choose scenario from selection
        #open SUMO with visualisation
        sumo_cmd = ["sumo-gui", "-c", sumo_config]
        self.sumo_process = traci.start(sumo_cmd)
        #important for vehicles not to use SUMO lane change model and start when defined
        for i in range(2):
            traci.vehicle.setLaneChangeMode(f"veh{i}", 0b000000000000)
        #reset all variables
        self.steps = 0       
        self.change_time = {}
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.variable_action_history = []
        self.lane_change_time = 0
        self.Time_to_main = 0
        self.collision = False



    def step(self):
        #important function that applies all actions and advances simulation
        agent_ids=self.identify_agents()
        if agent_ids["lane_changer"]==False: #if no LV no game to be considered
            traci.simulationStep()  #advance the simulation by one step
            self.steps += 1
            
        
        else:
            
            
            variable = {agent: self.get_variables(agent_ids[agent], agent_ids[self.agents[(idx+1)%2]], agent_ids["front_vehicle"]) for idx,
                     agent in enumerate(self.agents)}
        
            actions = self.get_actions(variable, agent_ids["lane_changer"])
            #store the variable and actions in history
            self.variable_action_history.append((variable, actions))
            
            for agent in self.agents:
                if agent == "lane_changer":
                    vehicle_id = agent_ids["lane_changer"]
                    other_vehicle_id = agent_ids["give_way"]
                    lane_changer_action = actions["lane_changer"]
                    
                    #slow LV down as approach junction to give space if non there
                    if variable["lane_changer"][3]<100:
                        factor = max(variable["lane_changer"][3],0.5)
                        traci.vehicle.slowDown(vehicle_id, variable["lane_changer"][0]*factor, 0.1)
                        
                    if lane_changer_action == 1:
                        #set time for lane change after first commited
                        if vehicle_id not in self.change_time:                    
                            delay_steps = 20  #number of steps to delay the lane change
                            self.change_time[vehicle_id] = self.steps + delay_steps  # Schedule the lane change
                        
            
    
    
                elif agent == "give_way":
                    if agent_ids["give_way"]==False: #no actions if no agent
                        pass
                    else:
                        vehicle_id = agent_ids["give_way"]
                        other_vehicle_id = agent_ids["lane_changer"]
                        give_way_action = actions["give_way"]
                            
        
                        if give_way_action == 1:  #apply give way
                            self.give_way(vehicle_id,other_vehicle_id,2,2)
        
        
            if len(self.change_time)>0: 
                for vehicle_id, scheduled_step in list(self.change_time.items()):
                    if self.steps >= scheduled_step:
                        x = traci.vehicle.getDistance(vehicle_id)
                        lane = '1to1b_1' #only will work on this junction
                        traci.vehicle.moveTo(vehicle_id, lane, x) #apply lane change at given time
                        if agent_ids["front_vehicle"]==False:
                            pass
                        else:
                            self.give_way(vehicle_id,agent_ids["front_vehicle"],2,1) #make sure doesn't instantly crash with FV
                        #collect times for rewards
                        self.lane_change_time = self.steps/10
                        if agent_ids["give_way"]==False:
                            self.Time_to_main=100
                        else:
                            self.Time_to_main = self.lane_change_time + variable["give_way"][3]/variable["give_way"][0]
                        
                        #check if lane change occurs
                        print(f"Lane change implemented for {vehicle_id} at step {self.steps}")
                    
                        # remove the vehicle from the schedule after lane change
                        del self.change_time[vehicle_id]
    
            traci.simulationStep()  #advance the simulation by one step
            colliding_vehicles = traci.simulation.getCollidingVehiclesIDList()
            if len(colliding_vehicles)>0:
                self.collision=True
                print('collsion')
            self.steps += 1
            
        
        

    
    def apply_final_rewards(self,Lane_change_time,Time_to_main):
        #extract values required for reward calculation from the simulation
        t_RV_cur = Time_to_main
        t_LV_cur = Lane_change_time
    
        #list to store rewards for each variable-action pair
        reward_history = []
        for variables, actions in self.variable_action_history:
            
            rewards = self.calculate_rewards(actions['lane_changer'], actions['give_way'],
                variables['lane_changer'], variables['give_way'],t_LV_cur, t_RV_cur)
            
            
            reward_history.append((variables, actions, rewards))
        
        #apply rewards to each variable-action pair
        
        for variables, actions, rewards in reward_history:
            for agent in self.agents:
                
                full_var = variables[agent]
                max_game_distance = full_var[8]*4 #4seconds x max speed
                if agent == "lane_changer": #make sure values fall within bins
                    var_1 = min(full_var[5],max_game_distance)*min(full_var[6],max_game_distance)
                    var_2 = min(abs(full_var[4]),full_var[8])*np.sign(full_var[4])
                    if var_1>(full_var[8]*2)**2:
                        var_1 = self.max_var_value_1[agent]
                    if var_1<0:
                        var_1 = self.min_var_value_1[agent]


                if agent == "give_way": #make sure values fall within bins
                    var_1 = full_var[3]/full_var[2]
                    var_2 = full_var[4]+full_var[5]
                    if var_1>full_var[8]:
                        var_1=self.max_var_value_1[agent]
                    if abs(var_2)>max_game_distance:
                        var_2 = self.max_var_value_2[agent]*np.sign(var_2)
                self.rewards[agent]+=rewards[agent] #summation of rewards over episode to study learning
                self.policy_managers[agent].update_policy(var_1,var_2,actions[agent],rewards[agent],0.3) #apply rewards
                
        




class PolicyManager:
    def __init__(self, num_bins_1, num_bins_2, min_value_1, max_value_1, min_value_2, max_value_2):
        #initiate matrix
        self.num_bins_1 = num_bins_1
        self.num_bins_2 = num_bins_2
        self.min_value_1 = min_value_1
        self.max_value_1 = max_value_1
        self.min_value_2 = min_value_2
        self.max_value_2 = max_value_2
        self.bin_width_1 = (max_value_1-min_value_1)/num_bins_1
        self.bin_width_2 = (max_value_2-min_value_2)/num_bins_2
        self.policies = np.empty((num_bins_1, num_bins_2, 2))

        #assign starting values for policy
        self.policies[:, :, 0] = 0.95 # assign chance of 0 action
        self.policies[:, :, 1] = 0.05  # assign chance of 1 action

    def get_policy(self, var_1, var_2):
        #pick out correct poicy
        bin_index_1 = min(int((var_1-self.min_value_1) / self.bin_width_1), self.num_bins_1 - 1)
        bin_index_2 = min(int((var_2-self.min_value_2) / self.bin_width_2), self.num_bins_2 - 1)
        return self.policies[bin_index_1, bin_index_2]

    def update_policy(self, var_1, var_2, action, reward, alpha):
        policy=self.get_policy(var_1,var_2)
            
        #apply Cross Learning
        for i in range(len(policy)):
            if i == action:
                policy[i] += alpha * reward * (1 - policy[i])
                policy[i]=round(policy[i],7) #round to help sum to 1
                
            else:
                policy[i] -= alpha * reward * policy[i]
                policy[i]=round(policy[i],7) #round to help sum to 1
            
            #don't fall out of range of probability
            if policy[i]>1:
                policy[i]=1
            if policy[i]<0:
                policy[i]=0
        
        
        policy_sum = sum(policy)
        
        if policy_sum != 1:  #only divide of policy does not sum to 1
            
            policy = [p / policy_sum for p in policy]
            
        
            









