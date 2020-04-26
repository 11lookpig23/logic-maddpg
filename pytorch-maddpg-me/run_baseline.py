#from ddpg3 import RL_agent
import tensorflow as tf
import numpy as np
import gym
import time
from envs import env1
import random
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 3000
MAX_EP_STEPS = 160
height = 8
ag_num = 5
env = env1.Lift(ag_num,height)#gym.make(ENV_NAME)
agents = []
agents_pg = []
#for i in range(ag_num):
#    agents.append(RL_agent(height,i))

for i in range(ag_num):
    RL = PolicyGradient(
        n_actions= 3,#env.action_space.n,
        n_features= 4*height+1, #env.observation_space.shape[0],
        learning_rate=0.004,
        reward_decay=0.9995,
        id = i,
        # output_graph=True,
    )
    agents_pg.append(RL)


def run_ddpg():
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        #s = np.array(s[0])
        ep_reward = np.array([0]*ag_num)

        for j in range(MAX_EP_STEPS):
            acts = []
            for ag in range(ag_num):
                a,a_prob = agents[ag].actor.choose_action(np.array(s[ag]))
                acts.append(a)
            s_, r, done, info = env.step(acts)
            for ag in range(ag_num):
                agents[ag].run_ep(s_[ag], r[ag], done[ag], info['new_act'][ag],s[ag])
            s = s_
            ep_reward  = ep_reward+np.array(r)
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: ', ep_reward)
    return

def run_ddpg2():
    t1 = time.time()
    final_rw = []
    avgres = []
    for i in range(MAX_EPISODES):
        #s = env.reset()
        s2 = env.reset()
        #s = np.array(s[0])
        ep_reward = np.array([0]*ag_num)
        ep_rw_all = []
        for j in range(MAX_EP_STEPS):
            #acts = []
            actionrl = []
            for ag in range(ag_num):
                #a,a_prob = agents[ag].actor.choose_action(np.array(s[ag]))
                actionrl.append(agents_pg[ag].choose_action(s2[ag]))
                #acts.append(a)
            #s_, r, done, info = env.step(acts)
            s_2, r2, done2, info2 = env.step(actionrl)
            for ag in range(ag_num):
            #ag = 0
            #print(" s .. a .. r ",s[0][-1],info['new_act'][0],r[0])
            #agents[ag].run_ep(s_[ag], r[ag], done[ag], info['new_act'][ag],s[ag])
                agents_pg[ag].store_transition(s2[ag], info2['new_act'][ag], r2[ag])
            #s = s_
            s2 = s_2
            #ep_reward  = ep_reward+np.array(r)
            #if j == MAX_EP_STEPS-1:
            #    print('Episode:', i, ' Reward: ', ep_reward)
        for ag in range(ag_num):
            ep_rs_sum = sum(agents_pg[ag].ep_rs)
            ep_rw_all.append(ep_rs_sum)
            vt = agents_pg[ag].learn()
        if i%5 == 0:
            print("episode-rl-pg:", i, "  reward:", ep_rw_all, ' sum_reward ',sum(ep_rw_all))
        final_rw.append(sum(ep_rw_all)/(ag_num*MAX_EP_STEPS))
        if (i>99 and i%100 == 0):
            np.save('pg_rw.npy',np.array(final_rw))
            print("avg ---- ",sum(final_rw[i-100:i])/100)
            avgres.append(sum(final_rw[i-100:i])/100)
            print(avgres)
    return avgres
avgres = run_ddpg2()
plt.plot( range(len(avgres)),avgres )
plt.show()