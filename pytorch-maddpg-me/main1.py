#from madrl_environments.pursuit import MAWaterWorld_mod
from envs import env1
from MADDPG import MADDPG
import numpy as np
import torch as th
#import visdom
from params import scale_reward
import matplotlib.pyplot as plt
from DILP import dilp_test
from collections import Counter
# do not render the scene
e_render = False
n_coop = 2


def f_s_b(pos,grid):
    front = False
    side = False
    back = False
    #pos,gird = trans_obs1(height,obs1)
    x = int(pos[0])
    y = int(pos[1])
    try:
        if grid[x+1,y]==1:
            front = True
    except:
        front = False
    if grid[x,int(1-y)]==1:
        side = True
    try:
        if grid[x-1,y]==1:
            back = True
    except:
        back = False
    return front,side,back

def trans_obs1(height,obs):
    ## obs: self-pos + other-pos + busy
    s1 = obs[:height*2]
    s2 = obs[height*2:4*height]
    busy = obs[-1]
    selfpos = s1.reshape((height,2))
    otherpos = s2.reshape((height,2))
    grid  = selfpos+otherpos
    pos = np.where(selfpos==1)
    return pos,grid

def getBGlist(height,index,total):
    bg = []
    for j in range(total):
        obs = world.reset()
        #index = 1
        obs1 = th.from_numpy(obs[index]).type(th.FloatTensor)
        obs = np.stack(obs)
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        #actout = model(obs1)
        action = maddpg.select_action(obs).data.cpu()
        #actout = action.detach().numpy()[index]
        actout = maddpg.actors[1](obs1).detach().numpy()
       # print("actput???????",actout)
        pos,grid = trans_obs1(height,obs1)
        f,s,b = f_s_b(pos,grid)
       # print(" busy? ",obs[index][-1]," f,s,b ", f,s,b)
       # print(" action -- ",actout)
        npact = actout#.detach().numpy()
        action0 = np.argmax(npact)
       # print("action0----",action0)
        prop = []
        if f==True:
            prop.append("front")
        elif s == True:
            prop.append("side")
        elif b == True:
            prop.append("back")
        bg.append((obs[index][-1],prop,action0))
    return bg

def rule_learning(mode = "up"):
    ### create data

    bg0 = getBGlist(height,1,400)
    bg1 = getBGlist(height,0,200)
    bg_all = bg0+bg1
    #np.save("bg_torch_2.npy",bg_all)
    #bg_all = np.load("bg_torch_3.npy",allow_pickle= True)
    if mode == "up":
        loss, cl_set = dilp_test.action_up(bg_all)
    elif mode == "switch":
        loss,cl_set = dilp_test.action_sw(bg_all)
    else:
        loss,cl_set = dilp_test.action_stay(bg_all)
    print(loss,"loss0000000")
    if loss[-1]>0.005:
        return []
    return cl_set

def rule_decoder(clause):
    head = clause.head
    body = clause.body
    if head.predicate == "up":
        move = 2
    elif head.predicate == "switch":
        move = 1
    else:
        move = 0
    atomli = []
    for b in body:
        atomli.append(b.predicate)
    return atomli,move

def make_rule(atomli,tar,obs1):
    modes = []
    pos,grid = trans_obs1(height,obs1)
    for atom in atomli:
        if atom=="busy":
            mode = (obs1[-1]==1)
        elif atom=="front":
            try:
                mode = (grid[pos[0]+1,pos[1]] == 1)
            except:
                return None
        elif atom=="back":
            try:
                mode = (grid[pos[0]-1,pos[1]] == 1)
            except:
                return None
        elif atom=="side":
            mode = ( grid[pos[0],int(1-pos[1])] ==1 )
        elif atom == "NObusy":
            mode = (obs1[-1]==0)
        else:
            print(atom,"atom~~~~~")
            mode = "None!!"
        modes.append(mode)
    if modes[0]==True:
        return tar
    else:
        return None


def learn_rules(mode = "up"):
    cl_set1 = rule_learning(mode)
    if cl_set1==[]:
        print("none,none.........................")
        return None,None
    cl_set2 = rule_learning(mode)
    cl_set3 = rule_learning(mode)
    # cl_set4 = rule_learning(height)
    cl_set = cl_set1+cl_set2+cl_set3#+cl_set4
    print("cl..................",cl_set)
    ats = []
    for cl1 in cl_set:
        atomli,tar = rule_decoder(cl1)
        ats=ats+atomli
    print("acts.....",ats)
    counter = Counter(ats)
    print(counter)
    com = counter.most_common()
    bodys = [com[0][0]]
    target = tar
    print("rules..........",bodys[0],"target......",target)
    return target,bodys ## up,[busy]


def applyRules(target,bodys,obs1):
    modes = []
    pos,grid = trans_obs1(height,obs1)
    for atom in bodys:
        if atom=="busy":
            mode = (obs1[-1]==1)
        elif atom=="front":
            try:
                mode = (grid[pos[0]+1,pos[1]] == 1)
            except:
                return None
        elif atom=="back":
            try:
                mode = (grid[pos[0]-1,pos[1]] == 1)
            except:
                return None
        elif atom=="side":
            mode = ( grid[pos[0],int(1-pos[1])] ==1 )
        elif atom == "NObusy":
            mode = (obs1[-1]==0)
        else:
            print(atom,"atom~~~~~")
            mode = "None!!"
        modes.append(mode)
    if modes[0]==True:
        #print(" modes[0]...",obs1[-1],"target_move...",target)
        return target
    else:
        return None 


n_agents = 5
height = 15
world = env1.Lift(n_agents,height)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
#world.seed(1234)

n_states = 4*height+1
n_actions = 3
capacity = 1000000
batch_size = 128

n_episode = 90#120#0#00
max_steps = 150#0
episodes_before_train = 25

win = None
param = None
rwli = []
rule_decode = False
maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    if (i_episode==41):
        #target = 2
        #bodys = ["busy"]
        target,bodys = None,None
        target1,bodys1 = learn_rules("up")
        target2,bodys2 = learn_rules("switch")
        target3,bodys3 = learn_rules("stay")
        for tb in [(target1,bodys1),(target2,bodys2),(target3,bodys3)]:
            if tb[0]!=None and tb[1]!=None:
                target,bodys = tb[0],tb[1]
        print("learn?????",target,bodys)
        if target==None and bodys == None:
            break
        rule_decode = True
    obs = world.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))

    for t in range(max_steps):
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
       # print(action.size())
        actli = [np.argmax(x) for x in action.numpy()]
        #rules = False
        if rule_decode == True:
            for i in range(n_agents):
                rule_act = applyRules(target,bodys,obs[i])
                if rule_act!= None:
                    actli[i] = target
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
    np.save('maddpg_ag5h15_with_r1.npy',np.array(rwli))
    if i_episode%30 == 0:
        print(rwli)
print(rwli)
np.save('maddpg_ag5h15_with_r1.npy',np.array(rwli))
plt.plot( range(len(rwli)),rwli )
plt.savefig("maddpg_ag5h15_with_r1.png")
plt.show()