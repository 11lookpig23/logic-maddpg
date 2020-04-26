
import numpy as np
import tensorflow as tf
import gym
import time
from envs import env1
import random
def rule_agent(height,obs):
    ## obs: self-pos + other-pos + busy
    s1 = obs[:height*2]
    s2 = obs[height*2:4*height]
    busy = obs[-1]
    selfpos = s1.reshape((height,2))
    otherpos = s2.reshape((height,2))
    grid  = selfpos+otherpos
    #x = self.pos[0]
    #y = self.pos[1]
    # print( " grid ",grid)
    pos = np.where(selfpos==1)
    try:
        x = int(pos[0])
        y = int(pos[1])
    except:
        return 0
    
    # print(" pos ",x,y)
    
    if busy == 1:
        if x==len(grid)-1:
            action = 2
            return action
        if (grid[x+1,y]==1 and  y == 1):
            action = 1
        else:
            action = 2
    else:
        if y==0:
            action = 1
        else:
            action = 0
    return action


def run_rule():
    t1 = time.time()
    rw_all = 0
    for i in range(MAX_EPISODES):
        s = env.reset()
        #s = np.array(s[0])
        ep_reward = np.array([0]*ag_num)

        for j in range(MAX_EP_STEPS):
            acts = []
            for ag in range(ag_num):
                #a = rule_agent(height,s[ag])#agents[ag].actor.choose_action(np.array(s[ag]))
                a = random.choice((0,1,2))
                acts.append(a)
            s_, r, done, info = env.step(acts)
            #for ag in range(ag_num):
            #    agents[ag].run_ep(s_[ag], r[ag], done[ag], info['new_act'][ag],s[ag])
            s = s_
            ep_reward  = ep_reward+np.array(r)
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: ',sum(ep_reward))
        if (i>60):
            rw_all += sum(ep_reward)/(len(ep_reward)*MAX_EP_STEPS)
    return  rw_all/(MAX_EPISODES-60)

MAX_EPISODES = 100#3000
MAX_EP_STEPS = 102
height = 8
ag_num = 5
env = env1.Lift(ag_num,height)#gym.make(ENV_NAME)
val = run_rule()
print(val)