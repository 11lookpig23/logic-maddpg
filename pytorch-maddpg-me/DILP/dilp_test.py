'''For testing DILP
'''
import sys 
sys.path.append("DILP/")#/..") 
from src.core import Term, Atom
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.dilp import DILP
import tensorflow as tf
import random
import numpy as np
tf.enable_eager_execution()


def getB(bg):
    B =[]
    i = 0
    for lg in bg:
        if lg[0] == 1:
            B.append(("busy",i))
        else:
            B.append(("NObusy",i))
        for k in lg[1]:
            B.append((k,i))
        i += 1
    B
    return B

def get_PN(bg_all):
    P_up = []
    P_sw = []
    P_stay = []
    N_up = []
    N_sw = []
    N_stay = []
    def P_N(bg):
        j = 0
        for lg in bg:
            if lg[2]==2:
                P_up.append(j)
            else:
                N_up.append(j)
            if lg[2]==1:
                P_sw.append(j)
            else:
                N_sw.append(j)
            if lg[2]==0:
                P_stay.append(j)
            else:
                N_stay.append(j)
            j+=1
    P_N(bg_all)
    return P_up,P_sw,P_stay,N_up,N_sw,N_stay

def get_B_N_P(bg_all):
    #bg_all = np.load("bg_all.npy",allow_pickle = True)
    B = getB(bg_all)
    P_up,P_sw,P_stay,N_up,N_sw,N_stay = get_PN(bg_all)
    return B,P_up,P_sw,P_stay,N_up,N_sw,N_stay


def action_up(bg_all):
    B_0,P_up,P_sw,P_stay,N_up,N_sw,N_stay = get_B_N_P(bg_all) 
    B = [Atom([Term(False, str(x[1]))], x[0]) for x in B_0]

    P =  [Atom([Term(False, str(id))], 'up')
         for id in P_up]
    N = [Atom([Term(False, str(id))], 'up')
         for id in N_up]
    term_x_0 = Term(True, 'X')
    term_x_1 = Term(True, 'X1')
    p_e = [Atom([term_x_0], 'busy'), Atom([term_x_0], 'NObusy'),Atom([term_x_0], 'side'),Atom([term_x_0], 'front'),Atom([term_x_0], 'back')]
    p_a = []

    target = Atom([term_x_0], 'up')
    # target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    target_rule = (Rule_Template(0, False), None)
    rules = {target: target_rule}
    constants = [str(i) for i in range(0, 400)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    dilp = DILP(langage_frame, B, P, N, program_template)
    loss,cl_set = dilp.train(steps = 301)
    return loss,cl_set

def action_up_2():
    B_0,P_up,P_sw,P_stay,N_up,N_sw,N_stay = get_B_N_P() 
    B = [Atom([Term(False, str(x[1]))], x[0]) for x in B_0]

    N =  [Atom([Term(False, str(id))], 'up')
         for id in P_up]
    P = [Atom([Term(False, str(id))], 'up')
         for id in N_up]
    term_x_0 = Term(True, 'X')
    term_x_1 = Term(True, 'X1')
    p_e = [Atom([term_x_0], 'busy'), Atom([term_x_0], 'NObusy'),Atom([term_x_0], 'side'),Atom([term_x_0], 'front'),Atom([term_x_0], 'back')]
    p_a = []

    target = Atom([term_x_0], 'up')
    # target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    target_rule = (Rule_Template(0, False), None)
    rules = {target: target_rule}
    constants = [str(i) for i in range(0, 400)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    dilp = DILP(langage_frame, B, P, N, program_template)
    dilp.train(steps = 250)


def action_switch():
    #num = 70
    #inx=random.sample(range(0,100),num)
    #neg_inx = list(set(range(100))-set(inx))
    B_0,P_up,P_sw,P_stay,N_up,N_sw,N_stay = get_B_N_P() 
    B = [Atom([Term(False, str(x[1]))], x[0]) for x in B_0]
   # B = [Atom([Term(False, str(id))], 'busy')
   #      for id in inx]+ [Atom([Term(False, str(id))], 'NObusy')
   #      for id in neg_inx]
   # print("B..........")
   # print(B)
    P =  [Atom([Term(False, str(id))], 'sw')
         for id in P_sw]
    N = []#[Atom([Term(False, str(id))], 'sw')
         #for id in N_sw]
   # print("P..........")
   # print(P)
   # print("N..........")
   # print(N)
    term_x_0 = Term(True, 'X')
    term_x_1 = Term(True, 'X1')
    p_e = [Atom([term_x_0], 'busy'), Atom([term_x_0], 'NObusy'),Atom([term_x_0], 'side'),Atom([term_x_0], 'front'),Atom([term_x_0], 'back')]
    #p_a = []
    p_a = [Atom([term_x_0], 'pred')]
    target = Atom([term_x_0], 'sw')
    # target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    target_rule = (Rule_Template(0, True), Rule_Template(0, False))
    p_a_rule = (Rule_Template(0, False), None)
    rules = {p_a[0]: p_a_rule,target: target_rule}
    constants = [str(i) for i in range(0, 400)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    dilp = DILP(langage_frame, B, P, N, program_template)
    dilp.train(steps = 252)


def action_switch_3():
    #num = 70
    #inx=random.sample(range(0,100),num)
    #neg_inx = list(set(range(100))-set(inx))
    B_0,P_up,P_sw,P_stay,N_up,N_sw,N_stay = get_B_N_P() 
    B = [Atom([Term(False, str(x[1]))], x[0]) for x in B_0]
    P =  [Atom([Term(False, str(id))], 'sw')
         for id in P_sw]
    N = []
    term_x_0 = Term(True, 'X')
    term_x_1 = Term(True, 'X1')
    p_e = [Atom([term_x_0], 'busy'), Atom([term_x_0], 'NObusy'),Atom([term_x_0], 'side'),Atom([term_x_0], 'front'),Atom([term_x_0], 'back')]
    p_a = []
    target = Atom([term_x_0], 'sw')
    # target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    target_rule = (Rule_Template(0, False), None)
    rules = {target: target_rule}
    constants = [str(i) for i in range(0, 400)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    dilp = DILP(langage_frame, B, P, N, program_template)
    dilp.train(steps = 252)

def action_sw(bg_all):
    #num = 70
    #inx=random.sample(range(0,100),num)
    #neg_inx = list(set(range(100))-set(inx))
    B_0,P_up,P_sw,P_stay,N_up,N_sw,N_stay = get_B_N_P(bg_all) 
    B = [Atom([Term(False, str(x[1]))], x[0]) for x in B_0]
    P =  [Atom([Term(False, str(id))], 'sw')
         for id in P_sw]
    N =  [Atom([Term(False, str(id))], 'sw')
         for id in N_sw]
    term_x_0 = Term(True, 'X')
    term_x_1 = Term(True, 'X1')
    p_e = [Atom([term_x_0], 'busy'), Atom([term_x_0], 'NObusy'),Atom([term_x_0], 'side'),Atom([term_x_0], 'front'),Atom([term_x_0], 'back')]
    p_a = []
    target = Atom([term_x_0], 'sw')
    # target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    target_rule = (Rule_Template(0, False), None)
    rules = {target: target_rule}
    constants = [str(i) for i in range(0, 400)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    dilp = DILP(langage_frame, B, P, N, program_template)
    loss,cl_set = dilp.train(steps = 51)
    return loss,cl_set

def action_stay(bg_all):
    B_0,P_up,P_sw,P_stay,N_up,N_sw,N_stay = get_B_N_P(bg_all) 
    B = [Atom([Term(False, str(x[1]))], x[0]) for x in B_0]

    P =  [Atom([Term(False, str(id))], 'stay')
         for id in P_stay]
    N = [Atom([Term(False, str(id))], 'stay')
         for id in N_stay]
    term_x_0 = Term(True, 'X')
    term_x_1 = Term(True, 'X1')
    p_e = [Atom([term_x_0], 'busy'), Atom([term_x_0], 'NObusy'),Atom([term_x_0], 'side'),Atom([term_x_0], 'front'),Atom([term_x_0], 'back')]
    p_a = []

    target = Atom([term_x_0], 'stay')
    # target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    target_rule = (Rule_Template(0, False), None)
    rules = {target: target_rule}
    constants = [str(i) for i in range(0, 400)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    dilp = DILP(langage_frame, B, P, N, program_template)
    loss,cl_set = dilp.train(steps = 51)
    return loss,cl_set

def action_switch_2():
    num = 30
    inx=random.sample(range(0,100),num)
    neg_inx = list(set(range(100))-set(inx))
    B = [Atom([Term(False, str(id))], 'busy')
        for id in inx]+ [Atom([Term(False, str(id))], 'NObusy')
         for id in neg_inx]+ [Atom([Term(False, str(id))], 'front')
         for id in neg_inx if id%2==0]+[Atom([Term(False, str(id))], 'back')
         for id in neg_inx if id%2!=0]
    P =  [Atom([Term(False, str(id))], 'switch')
         for id in neg_inx]
    N = [Atom([Term(False, str(id))], 'switch')
         for id in inx]
    term_x_0 = Term(True, 'X')
    term_x_1 = Term(True, 'X1')
    p_e = [Atom([term_x_0], 'busy'), Atom([term_x_0], 'NObusy'),Atom([term_x_0], 'side'),Atom([term_x_0], 'front'),Atom([term_x_0], 'back')]
    p_a = []
    target = Atom([term_x_0], 'switch')
    # target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    target_rule = (Rule_Template(0, False), None)
    rules = {target: target_rule}
    constants = [str(i) for i in range(0, 100)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    dilp = DILP(langage_frame, B, P, N, program_template)
    dilp.train(steps = 252)




#prdecessor()

if __name__ == '__main__':
    action_sw()
    