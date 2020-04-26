import numpy as np
import sys 
sys.path.append("../..") 
from malib.spaces import Discrete, Box, MASpace, MAEnvSpec
from malib.environments.base_game import BaseGame
from malib.error import EnvironmentNotFound, WrongNumberOfAgent, WrongActionInputLength
import random
from multiagent.multi_discrete import MultiDiscrete
import copy
from malib.environments.gridworld import Grid

class Lift(BaseGame):
    def __init__(self,agent_num,height):
        self.agent_num = agent_num
        self.height = height
        obs_dim = height*4+1
        self.action_range = [0, 1, 2]
        self.action_spaces = MASpace(
    tuple(Box(low = 0 , high=2, shape=(1,), dtype=np.int32) for _ in range(self.agent_num)))
        self.observation_spaces = MASpace(
            tuple(Box(
                    low = 0 , high=1, shape=(obs_dim,), dtype=np.int32
                ) for _ in range(self.agent_num))
        )
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.grid = []
        self.busy_n = {}

    def get_game_list(self):
        pass

    def get_rewards(self):
        pass

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = [False]*self.agent_num
        info_n = {"new_act":[]}

        grid1,busy_n,obs_n,true_act_n,reward_n = self.elevator.run_a_step(action_n)
        self.grid = grid1
        self.busy_n = busy_n
        info_n["new_act"] = true_act_n
        ## change grid
        ## new action
        return obs_n, reward_n, done_n, info_n
    def trans_obs(self,pos):
        shape =  self.grid.shape[0]*self.grid.shape[1]
        selfpos = np.zeros((self.grid.shape[0],self.grid.shape[1]))
        selfpos[pos] = 1
        selfpos = np.reshape(selfpos,shape)
        grid1 = copy.deepcopy(self.grid)
        grid1[pos] = 0
        otherpos = np.reshape(grid1,shape)
        inputarr = np.hstack((selfpos,otherpos))
        return inputarr
    def reset(self):
        ## obser_n
        ## 只是部分的obs
        arr = np.zeros(self.height*2)
        num = self.agent_num#self.height-3
        inx=random.sample(range(0,self.height*2),num)
        arr[inx] = 1
        arr = np.reshape(arr,(self.height,2))
        self.grid = arr
        obs_n = []
        namelist = []
        k2 = 1
        for i in range(self.height):
            for j in range(2):
                if self.grid[i,j]!=0:
                    inputarr0 = self.trans_obs((i,j))
                    if(k2%2==0):
                        busy = 0
                    else:
                        busy = 1
                    self.busy_n["Ag"+str(k2)] = busy
                    obs_n.append(np.hstack((inputarr0, np.array([busy]))))
                    namelist.append("Ag"+str(k2))
                    k2+=1
        ### create a grid world
        ## combine grid and busy
        print(" namelist +++ ",namelist)
        #print( " busy_n dict ", self.busy_n )
        #print(" == Grid == ",self.grid)
        self.elevator = Grid(self.grid,self.busy_n,self.agent_num,namelist)
        return obs_n
    
    def terminate(self):
        pass

    def render(self):
        print("ohhhhhhh")

if __name__ == "__main__":
    lift = Lift(10,6)
    lift.reset()
    action_space = lift.env_specs.action_space[3]
    #print("  action_space  ",action_space.shape)
    lift.step(np.array([2,1,0,1,1,2,2,2,2,2]))
    lift.step(np.array([1,2,2,1,0,2,1,0,2,1]))
