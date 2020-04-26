"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time
from envs import env1
import random

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 2000
MAX_EP_STEPS = 102
LR_A = 0.00007    # learning rate for actor
LR_C = 0.00007   # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'

###############################  Actor  ####################################


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement,order):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0
        self.name = order

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 64, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1'+self.name,
                                  trainable=trainable)
            net2 = tf.layers.dense(net, 32, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2'+self.name,
                                  trainable=trainable)
            with tf.variable_scope('a'+self.name):
                actions = tf.layers.dense(net2, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a'+self.name, trainable=trainable)
                #scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
                a_prob = tf.nn.softmax(actions)
        return a_prob#scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        a_prob = self.sess.run(self.a, feed_dict={S: s})[0]  # single action
        rnum = random.randint(0, 1)
        #if rnum == 0:
        #    action = np.random.choice(range(self.a_dim),p = a_prob.ravel())
        #else:
        action = np.argmax(a_prob)
        return action,a_prob

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'+self.name):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'+self.name):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_,order):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement
        self.name = order

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)    # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'+self.name):
                n_l1 = 100
                w1_s = tf.get_variable('w1_s'+self.name, [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a'+self.name, [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1'+self.name, [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'+self.name):
                q = tf.layers.dense(net, 16, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
                q2 = tf.layers.dense(q, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)

        return q2

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

height = 6
ag_num = 2
env = env1.Lift(ag_num,height)#gym.make(ENV_NAME)
#env = env.unwrapped
#env.seed(1)
state_dim = 4*height+1#env.observation_space.shape[0]
action_dim = 3  #env.action_space.shape[0]
action_bound = 2#env.action_space.high

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()
#def run_ddpg

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
actorli = []
criticli = []
Mli = []
for i in range(ag_num):
    actorli.append(Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT,str(i)))
    criticli.append(Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actorli[-1].a, actorli[-1].a_,str(i)))
    Mli.append( Memory(MEMORY_CAPACITY, dims=2 * state_dim + 1 + 1) )
    actorli[-1].add_grad_to_graph(criticli[-1].a_grads)

sess.run(tf.global_variables_initializer())

#M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + 1 + 1)

#if OUTPUT_GRAPH:
#    tf.summary.FileWriter("logs/", sess.graph)

var = 3  # control exploration

t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    
    ep_reward = 0

    for j in range(MAX_EP_STEPS):

        #if RENDER:
        #    env.render()

        # Add exploration noise
        act_n = []
        for ag in range(ag_num):
            s0 = np.array(s[ag])
            
            a , a_prob= actorli[ag].choose_action(s0)
            act_n.append(a)
        #a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        
        s_, r, done, info = env.step(act_n)
        #print("s -----")
        #print(r)
        #print(s_)
        r_li = []
        #print( 'r....2 ',r)
        for ag in range(ag_num):
            #print(" r ..... ",r)
            #print(" s_ ..... ",s_)
            s0_, r_0 = np.array(s_[ag]), r[ag]#, done[ag]#, info['new_act'][0]
            a_new = info["new_act"][ag]
            r_li.append(r_0)
        #print(" s r a  ",s_,r,a_new)
            Mli[ag].store_transition(s[ag], a_new, r_0 / 10, s0_)

            if Mli[ag].pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                b_M = Mli[ag].sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_a = b_M[:, state_dim: state_dim + action_dim]
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]

                criticli[ag].learn(b_s, b_a, b_r, b_s_)
                actorli[ag].learn(b_s)

        s = s_
        #print(" r_li === ",r_li)
        ep_reward += sum(r_li)

        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward))#, 'Explore: %.2f' % var, )
            #if ep_reward > -300:
            #    RENDER = True
            break

print('Running time: ', time.time()-t1)

