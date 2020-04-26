#from madrl_environments.pursuit import MAWaterWorld_mod
from envs import env1
from MADDPG import MADDPG
import numpy as np
import torch as th
#import visdom
from params import scale_reward
import matplotlib.pyplot as plt
# do not render the scene
e_render = False
n_coop = 2

'''
food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
world = MAWaterWorld_mod(n_pursuers=2, n_evaders=50,
                         n_poison=50, obstacle_radius=0.04,
                         food_reward=food_reward,
                         poison_reward=poison_reward,
                         encounter_reward=encounter_reward,
                         n_coop=n_coop,
                         sensor_range=0.2, obstacle_loc=None, )

vis = visdom.Visdom(port=5274)
'''
world = env1.Lift(5,8)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
#world.seed(1234)
n_agents = 5#world.n_pursuers
n_states = 33
n_actions = 3
capacity = 1000000
batch_size = 128

n_episode = 100#0#00
max_steps = 160#0
episodes_before_train = 15

win = None
param = None
rwli = []
maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        #if i_episode % 100 == 0 and e_render:
        #    world.render()
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
       # print(action.size())
        actli = [np.argmax(x) for x in action.numpy()]
        obs_, reward, done, _ = world.step(actli)
        #print("Resear ..... ",reward)
        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)
    rwli.append(total_reward/(max_steps*n_agents))
    np.save('maddpgrw.npy',np.array(rwli))
print(rwli)
np.save('maddpgrw.npy',np.array(rwli))
plt.plot( range(n_episode),rwli )
plt.show()


'''
if maddpg.episode_done == maddpg.episodes_before_train:
    print('training now begins...')
    print('MADDPG on WaterWorld\n' +
            'scale_reward=%f\n' % scale_reward +
            'agent=%d' % n_agents +
            ', coop=%d' % n_coop +
            ' \nlr=0.001, 0.0001, sensor_range=0.3\n' +
            'food=%f, poison=%f, encounter=%f' % (
                food_reward,
                poison_reward,
                encounter_reward))
'''
'''
if win is None:
    win = vis.line(X=np.arange(i_episode, i_episode+1),
                    Y=np.array([
                        np.append(total_reward, rr)]),
                    opts=dict(
                        ylabel='Reward',
                        xlabel='Episode',
                        title='MADDPG on WaterWorld_mod\n' +
                        'agent=%d' % n_agents +
                        ', coop=%d' % n_coop +
                        ', sensor_range=0.2\n' +
                        'food=%f, poison=%f, encounter=%f' % (
                            food_reward,
                            poison_reward,
                            encounter_reward),
                        legend=['Total'] +
                        ['Agent-%d' % i for i in range(n_agents)]))
else:
    vis.line(X=np.array(
        [np.array(i_episode).repeat(n_agents+1)]),
                Y=np.array([np.append(total_reward,
                                    rr)]),
                win=win,
                update='append')

if param is None:
    param = vis.line(X=np.arange(i_episode, i_episode+1),
                        Y=np.array([maddpg.var[0]]),
                        opts=dict(
                            ylabel='Var',
                            xlabel='Episode',
                            title='MADDPG on WaterWorld: Exploration',
                            legend=['Variance']))
else:
    vis.line(X=np.array([i_episode]),
                Y=np.array([maddpg.var[0]]),
                win=param,
                update='append')
'''
#world.close()
