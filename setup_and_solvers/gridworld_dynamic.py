import itertools

import os, sys, getopt, pdb, string
import time

import pygame
# import pygame.locals as pgl
from pygame.locals import *

import numpy as np
# import random as pr
from random import choice
import random
from markov_decision_process import *


def wall_pattern(nrows, ncols, endstate=0, pattern="comb"):
    """Generate a specific wall pattern for a particular gridworld."""

    goal = coords(nrows, ncols, endstate)
    walls = []

    if goal[1] % 2 == 0:
        wmod = 1
    else:
        wmod = 0

    for i in range(ncols):
        for j in range(nrows):
            if i % 2 == wmod and j != nrows - 1:
                walls.append((i, j))

    return walls


class Gridworld():
    def __init__(self, current=0, nrows=8, ncols=8, robotmdp=MDP(), targets=[], obstacles=[], unsafe_states=[],
                 camp_sites=[], dynamic_agent_states=[]):
        # walls are the obstacles. The edges of the gridworld will be included into the walls.
        self.nrows = nrows
        self.ncols = ncols
        self.robotmdp = robotmdp
        self.nstates = nrows * ncols
        self.nactions = 4
        self.actlist = ['N', 'S', 'W', 'E']
        self.targets = targets
        self.left_edge = []
        self.right_edge = []
        self.top_edge = []
        self.bottom_edge = []
        self.obstacles = obstacles
        self.unsafe_states = unsafe_states
        self.camp_sites = camp_sites
        self.dynamic_agent_states = dynamic_agent_states
        self.dynamic_agent_states_dict = dict([])

        ind = 0
        for st in self.dynamic_agent_states:
            self.dynamic_agent_states_dict[ind] = st
            ind = ind + 1

        # self.states = [(x, y) for x in range(0, ncols) for y in range(0, nrows)]
        self.states = list([])
        self.states = [(x, y) for x in range(0, ncols) for y in range(0, nrows)]

        self.states_dict = dict([])
        indx = 0
        for x in range(0, ncols):
            for y in range(0, nrows):
                self.states_dict[indx] = (x, y)
                indx = indx + 1

        # prob = {a: np.zeros((self.nstates, self.nstates)) for a in self.actlist}
        # for s in self.states:
        #     for a in self.actlist:
        #         prob = self.getProbs(s, a, prob)

        self.mdp_states = list(itertools.product(range(self.nstates), self.dynamic_agent_states))
        self.mdp_states_dict = dict([])
        indx = 0
        for mdp_st in self.mdp_states:
            self.mdp_states_dict[indx] = mdp_st
            indx = indx + 1

        prob = {a: np.zeros((len(self.mdp_states), len(self.mdp_states))) for a in self.actlist}
        for s in self.mdp_states:
            for a in self.actlist:
                prob = self.getProbs(s, a, prob)

        self.mdp = MDP(current, self.actlist, range(len(self.mdp_states)))
        self.mdp.prob = prob

    def coords(self, s):
        return self.states[s]  # the coordinate for state s.

    def rcoord(self, x, y):
        return self.states.index((x, y))

    # def getProbs(self, state, action, prob):
    #     successors = []
    #     (x, y) = state
    #     s = self.states.index(state)
    #     if (s in self.obstacles) or (s in self.targets):
    #         prob[action][s, s] = 1
    #     else:
    #         if y - 1 >= 0:
    #             # south_state = (x, y - 1)
    #             west_state = (x, y - 1)
    #         else:
    #             # south_state = (x, y)
    #             west_state = (x, y)
    #         if y + 1 <= self.nrows - 1:
    #             # north_state = (x, y + 1)
    #             east_state = (x, y + 1)
    #         else:
    #             east_state = (x, y)
    #         if x - 1 >= 0:
    #             # west_state = (x - 1, y)
    #             north_state = (x - 1, y)
    #         else:
    #             # west_state = (x, y)
    #             north_state = (x, y)
    #         if x + 1 <= self.ncols - 1:
    #             # east_state = (x + 1, y)
    #             south_state = (x + 1, y)
    #         else:
    #             south_state = (x, y)
    #         successors.append((north_state, self.robotmdp.P(0, action, 1)))
    #         successors.append((west_state, self.robotmdp.P(0, action, 4)))
    #         successors.append((south_state, self.robotmdp.P(0, action, 3)))
    #         successors.append((east_state, self.robotmdp.P(0, action, 2)))
    #         for (next_state, p) in successors:
    #             ns = self.states.index(next_state)
    #             prob[action][s, ns] += p
    #     return prob

    # # The following is the normal NSEW dynamics - working with dynamic agent deterministic.
    # def getProbs(self, state, action, prob):
    #     successors = []
    #     (s, d) = state
    #     state_ind = self.mdp_states.index(state)
    #
    #     d_next = self.dynamic_agent_states.index(d) + 1
    #     if d_next >= len(self.dynamic_agent_states):
    #         d_next = self.dynamic_agent_states_dict[0]
    #     else:
    #         d_next = self.dynamic_agent_states_dict[d_next]
    #
    #     (x, y) = self.states_dict[s]
    #     # (x_d, y_d) = state[1]
    #     # s = self.states.index(state)
    #
    #     if (s in self.obstacles) or (s in self.targets) or (s in self.unsafe_states):
    #         prob[action][self.mdp_states.index(state), self.mdp_states.index(state)] = 1
    #     else:
    #         if y - 1 >= 0:
    #             # south_state = (x, y - 1)
    #             west_state = (x, y - 1)
    #             if self.states.index(west_state) in self.obstacles:
    #                 west_state = (x, y)
    #             else:
    #                 west_state = west_state
    #         else:
    #             # south_state = (x, y)
    #             west_state = (x, y)
    #         if y + 1 <= self.nrows - 1:
    #             # north_state = (x, y + 1)
    #             east_state = (x, y + 1)
    #             if self.states.index(east_state) in self.obstacles:
    #                 east_state = (x, y)
    #             else:
    #                 east_state = east_state
    #         else:
    #             east_state = (x, y)
    #         if x - 1 >= 0:
    #             # west_state = (x - 1, y)
    #             north_state = (x - 1, y)
    #             if self.states.index(north_state) in self.obstacles:
    #                 north_state = (x, y)
    #             else:
    #                 north_state = north_state
    #         else:
    #             # west_state = (x, y)
    #             north_state = (x, y)
    #         if x + 1 <= self.ncols - 1:
    #             # east_state = (x + 1, y)
    #             south_state = (x + 1, y)
    #             if self.states.index(south_state) in self.obstacles:
    #                 south_state = (x, y)
    #             else:
    #                 south_state = south_state
    #         else:
    #             south_state = (x, y)
    #         successors.append((north_state, self.robotmdp.P(0, action, 1)))
    #         successors.append((west_state, self.robotmdp.P(0, action, 4)))
    #         successors.append((south_state, self.robotmdp.P(0, action, 3)))
    #         successors.append((east_state, self.robotmdp.P(0, action, 2)))
    #         for (next_state, p) in successors:
    #             next_state_idx = self.states.index(next_state)
    #             ns_ind = self.mdp_states.index((next_state_idx, d_next))
    #             prob[action][state_ind, ns_ind] += p
    #     return prob

    # The following is for N, NE, NW dynamics.
    # def getProbs(self, state, action, prob):
    #     successors = []
    #     (x, y) = state
    #     s = self.states.index(state)
    #     if (s in self.obstacles) or (s in self.targets) or (s in self.unsafe_states):
    #         prob[action][s, s] = 1
    #     else:
    #         if action == 'N':
    #             if y - 1 >= 0:
    #                 # south_state = (x, y - 1)
    #                 west_state = (x - 1, y - 1)
    #                 if (x - 1 < 0) or (self.states.index(west_state) in self.obstacles):
    #                     west_state = (x, y)
    #                 else:
    #                     west_state = west_state
    #             else:
    #                 # south_state = (x, y)
    #                 west_state = (x, y)
    #             if y + 1 <= self.nrows - 1:
    #                 # north_state = (x, y + 1)
    #                 east_state = (x - 1, y + 1)
    #                 if (x-1 < 0) or (self.states.index(east_state) in self.obstacles):
    #                     east_state = (x, y)
    #                 else:
    #                     east_state = east_state
    #             else:
    #                 east_state = (x, y)
    #             if x - 1 >= 0:
    #                 # west_state = (x - 1, y)
    #                 north_state = (x - 1, y)
    #                 if self.states.index(north_state) in self.obstacles:
    #                     north_state = (x, y)
    #                 else:
    #                     north_state = north_state
    #             else:
    #                 # west_state = (x, y)
    #                 north_state = (x, y)
    #             if x + 1 <= self.ncols - 1:
    #                 # east_state = (x + 1, y)
    #                 south_state = (x + 1, y)
    #                 if self.states.index(south_state) in self.obstacles:
    #                     south_state = (x, y)
    #                 else:
    #                     south_state = south_state
    #             else:
    #                 south_state = (x, y)
    #         elif action == 'S':
    #             if y - 1 >= 0:
    #                 # south_state = (x, y - 1)
    #                 west_state = (x + 1, y - 1)
    #                 if (x + 1 > self.ncols-1) or (self.states.index(west_state) in self.obstacles):
    #                     west_state = (x, y)
    #                 else:
    #                     west_state = west_state
    #             else:
    #                 # south_state = (x, y)
    #                 west_state = (x, y)
    #             if y + 1 <= self.nrows - 1:
    #                 # north_state = (x, y + 1)
    #                 east_state = (x + 1, y + 1)
    #                 if (x + 1 > self.ncols-1) or (self.states.index(east_state) in self.obstacles):
    #                     east_state = (x, y)
    #                 else:
    #                     east_state = east_state
    #             else:
    #                 east_state = (x, y)
    #             if x - 1 >= 0:
    #                 # west_state = (x - 1, y)
    #                 north_state = (x - 1, y)
    #                 if self.states.index(north_state) in self.obstacles:
    #                     north_state = (x, y)
    #                 else:
    #                     north_state = north_state
    #             else:
    #                 # west_state = (x, y)
    #                 north_state = (x, y)
    #             if x + 1 <= self.ncols - 1:
    #                 # east_state = (x + 1, y)
    #                 south_state = (x + 1, y)
    #                 if self.states.index(south_state) in self.obstacles:
    #                     south_state = (x, y)
    #                 else:
    #                     south_state = south_state
    #             else:
    #                 south_state = (x, y)
    #         elif action == 'E':
    #             if y - 1 >= 0:
    #                 # south_state = (x, y - 1)
    #                 west_state = (x - 1, y - 1)
    #                 if (x - 1 < 0) or (self.states.index(west_state) in self.obstacles):
    #                     west_state = (x, y)
    #                 else:
    #                     west_state = west_state
    #             else:
    #                 # south_state = (x, y)
    #                 west_state = (x, y)
    #             if y + 1 <= self.nrows - 1:
    #                 # north_state = (x, y + 1)
    #                 east_state = (x , y + 1)
    #                 if (self.states.index(east_state) in self.obstacles):
    #                     east_state = (x, y)
    #                 else:
    #                     east_state = east_state
    #             else:
    #                 east_state = (x, y)
    #             if x - 1 >= 0:
    #                 # west_state = (x - 1, y)
    #                 north_state = (x - 1, y+1)
    #                 if (y+1 > self.nrows-1) or (self.states.index(north_state) in self.obstacles):
    #                     north_state = (x, y)
    #                 else:
    #                     north_state = north_state
    #             else:
    #                 # west_state = (x, y)
    #                 north_state = (x, y)
    #             if x + 1 <= self.ncols - 1:
    #                 # east_state = (x + 1, y)
    #                 south_state = (x + 1, y + 1)
    #                 if (y + 1 > self.nrows-1) or (self.states.index(south_state) in self.obstacles):
    #                     south_state = (x, y)
    #                 else:
    #                     south_state = south_state
    #             else:
    #                 south_state = (x, y)
    #
    #         elif action == 'W':
    #             if y - 1 >= 0:
    #                 # south_state = (x, y - 1)
    #                 west_state = (x, y - 1)
    #                 if (self.states.index(west_state) in self.obstacles):
    #                     west_state = (x, y)
    #                 else:
    #                     west_state = west_state
    #             else:
    #                 # south_state = (x, y)
    #                 west_state = (x, y)
    #             if y + 1 <= self.nrows - 1:
    #                 # north_state = (x, y + 1)
    #                 east_state = (x - 1, y + 1)
    #                 if (x - 1 < 0) or (self.states.index(east_state) in self.obstacles):
    #                     east_state = (x, y)
    #                 else:
    #                     east_state = east_state
    #             else:
    #                 east_state = (x, y)
    #             if x - 1 >= 0:
    #                 # west_state = (x - 1, y)
    #                 north_state = (x - 1, y - 1)
    #                 if (y - 1 < 0) or (self.states.index(north_state) in self.obstacles):
    #                     north_state = (x, y)
    #                 else:
    #                     north_state = north_state
    #             else:
    #                 # west_state = (x, y)
    #                 north_state = (x, y)
    #             if x + 1 <= self.ncols - 1:
    #                 # east_state = (x + 1, y)
    #                 south_state = (x + 1, y - 1)
    #                 if (y - 1 < 0) or (self.states.index(south_state) in self.obstacles):
    #                     south_state = (x, y)
    #                 else:
    #                     south_state = south_state
    #             else:
    #                 south_state = (x, y)
    #
    #         successors.append((north_state, self.robotmdp.P(0, action, 1)))
    #         successors.append((west_state, self.robotmdp.P(0, action, 4)))
    #         successors.append((south_state, self.robotmdp.P(0, action, 3)))
    #         successors.append((east_state, self.robotmdp.P(0, action, 2)))
    #         for (next_state, p) in successors:
    #             ns = self.states.index(next_state)
    #             prob[action][s, ns] += p
    #     return prob

    # Modified for the dynamic agent to have stochastic dynamics.
    def getProbs(self, state, action, prob):
        successors = []
        (s, d) = state
        state_ind = self.mdp_states.index(state)

        d_next_1 = self.dynamic_agent_states.index(d) + 1
        if d_next_1 >= len(self.dynamic_agent_states):
            d_next_1 = self.dynamic_agent_states_dict[0]
        else:
            d_next_1 = self.dynamic_agent_states_dict[d_next_1]

        d_next_2 = d
        # if d_next_2 >= len(self.dynamic_agent_states):
        #     d_next_2 = self.dynamic_agent_states_dict[0]
        # else:
        #     d_next_2 = self.dynamic_agent_states_dict[d_next_2]

        d_next = {d_next_1, d_next_2}

        (x, y) = self.states_dict[s]
        # (x_d, y_d) = state[1]
        # s = self.states.index(state)

        if (s in self.obstacles) or (s in self.targets) or (s in self.unsafe_states):
            prob[action][self.mdp_states.index(state), self.mdp_states.index(state)] = 1
        else:
            if y - 1 >= 0:
                # south_state = (x, y - 1)
                west_state = (x, y - 1)
                if self.states.index(west_state) in self.obstacles:
                    west_state = (x, y)
                else:
                    west_state = west_state
            else:
                # south_state = (x, y)
                west_state = (x, y)
            if y + 1 <= self.nrows - 1:
                # north_state = (x, y + 1)
                east_state = (x, y + 1)
                if self.states.index(east_state) in self.obstacles:
                    east_state = (x, y)
                else:
                    east_state = east_state
            else:
                east_state = (x, y)
            if x - 1 >= 0:
                # west_state = (x - 1, y)
                north_state = (x - 1, y)
                if self.states.index(north_state) in self.obstacles:
                    north_state = (x, y)
                else:
                    north_state = north_state
            else:
                # west_state = (x, y)
                north_state = (x, y)
            if x + 1 <= self.ncols - 1:
                # east_state = (x + 1, y)
                south_state = (x + 1, y)
                if self.states.index(south_state) in self.obstacles:
                    south_state = (x, y)
                else:
                    south_state = south_state
            else:
                south_state = (x, y)
            successors.append((north_state, self.robotmdp.P(0, action, 1)))
            successors.append((west_state, self.robotmdp.P(0, action, 4)))
            successors.append((south_state, self.robotmdp.P(0, action, 3)))
            successors.append((east_state, self.robotmdp.P(0, action, 2)))
            for (next_state, p) in successors:
                next_state_idx = self.states.index(next_state)
                if len(d_next) > 1:
                    for d_next_st in d_next:
                        ns_ind = self.mdp_states.index((next_state_idx, d_next_st))
                        if d_next_st == d_next_1:
                            prob[action][state_ind, ns_ind] += (p*0.65)
                        else:
                            prob[action][state_ind, ns_ind] += (p*0.35)
                else:
                    for d_next_st in d_next:
                        ns_ind = self.mdp_states.index((next_state_idx, d_next_st))
                        prob[action][state_ind, ns_ind] += (p*1)

        return prob

    # # Modified N, NE, NW dynamics. (does not bounce back to the same state.)
    # def getProbs(self, state, action, prob):
    #     successors = []
    #     (x, y) = state
    #     north_prob = True
    #     east_prob = True
    #     west_prob = True
    #     south_prob = True
    #     s = self.states.index(state)
    #     if (s in self.obstacles) or (s in self.targets) or (s in self.unsafe_states):
    #         prob[action][s, s] = 1
    #     else:
    #         if action == 'N':
    #             if y - 1 >= 0:
    #                 # south_state = (x, y - 1)
    #                 west_state = (x - 1, y - 1)
    #                 if (x - 1 < 0) or (self.states.index(west_state) in self.obstacles):
    #                     west_state = (x, y)
    #                     west_prob = False
    #                 else:
    #                     west_state = west_state
    #             else:
    #                 # south_state = (x, y)
    #                 west_state = (x, y)
    #                 west_prob  = False
    #
    #             if y + 1 <= self.nrows - 1:
    #                 # north_state = (x, y + 1)
    #                 east_state = (x - 1, y + 1)
    #                 if (x - 1 < 0) or (self.states.index(east_state) in self.obstacles):
    #                     east_state = (x, y)
    #                     east_prob = False
    #                 else:
    #                     east_state = east_state
    #             else:
    #                 east_state = (x, y)
    #                 east_prob  = False
    #
    #             if x - 1 >= 0:
    #                 # west_state = (x - 1, y)
    #                 north_state = (x - 1, y)
    #                 if self.states.index(north_state) in self.obstacles:
    #                     north_state = (x, y)
    #                     north_prob = False
    #                 else:
    #                     north_state = north_state
    #             else:
    #                 # west_state = (x, y)
    #                 north_state = (x, y)
    #                 north_prob  = False
    #
    #             if x + 1 <= self.ncols - 1:
    #                 # east_state = (x + 1, y)
    #                 south_state = (x + 1, y)
    #                 if self.states.index(south_state) in self.obstacles:
    #                     south_state = (x, y)
    #                     south_prob = False
    #                 else:
    #                     south_state = south_state
    #             else:
    #                 south_state = (x, y)
    #                 south_prob  = False
    #
    #         elif action == 'S':
    #             if y - 1 >= 0:
    #                 # south_state = (x, y - 1)
    #                 west_state = (x + 1, y - 1)
    #                 if (x + 1 > self.ncols - 1) or (self.states.index(west_state) in self.obstacles):
    #                     west_state = (x, y)
    #                     west_prob = False
    #                 else:
    #                     west_state = west_state
    #             else:
    #                 # south_state = (x, y)
    #                 west_state = (x, y)
    #                 west_prob  = False
    #
    #             if y + 1 <= self.nrows - 1:
    #                 # north_state = (x, y + 1)
    #                 east_state = (x + 1, y + 1)
    #                 if (x + 1 > self.ncols - 1) or (self.states.index(east_state) in self.obstacles):
    #                     east_state = (x, y)
    #                     east_prob = False
    #                 else:
    #                     east_state = east_state
    #             else:
    #                 east_state = (x, y)
    #                 east_prob  = False
    #
    #             if x - 1 >= 0:
    #                 # west_state = (x - 1, y)
    #                 north_state = (x - 1, y)
    #                 if self.states.index(north_state) in self.obstacles:
    #                     north_state = (x, y)
    #                     north_prob = False
    #                 else:
    #                     north_state = north_state
    #             else:
    #                 # west_state = (x, y)
    #                 north_state = (x, y)
    #                 north_prob  = False
    #
    #             if x + 1 <= self.ncols - 1:
    #                 # east_state = (x + 1, y)
    #                 south_state = (x + 1, y)
    #                 if self.states.index(south_state) in self.obstacles:
    #                     south_state = (x, y)
    #                     south_prob = False
    #                 else:
    #                     south_state = south_state
    #             else:
    #                 south_state = (x, y)
    #                 south_prob  = False
    #
    #         elif action == 'E':
    #             if y - 1 >= 0:
    #                 # south_state = (x, y - 1)
    #                 west_state = (x - 1, y - 1)
    #                 if (x - 1 < 0) or (self.states.index(west_state) in self.obstacles):
    #                     west_state = (x, y)
    #                     west_prob = False
    #                 else:
    #                     west_state = west_state
    #             else:
    #                 # south_state = (x, y)
    #                 west_state = (x, y)
    #                 west_prob  = False
    #
    #             if y + 1 <= self.nrows - 1:
    #                 # north_state = (x, y + 1)
    #                 east_state = (x, y + 1)
    #                 if (self.states.index(east_state) in self.obstacles):
    #                     east_state = (x, y)
    #                     east_prob = False
    #                 else:
    #                     east_state = east_state
    #             else:
    #                 east_state = (x, y)
    #                 east_prob  = False
    #
    #             if x - 1 >= 0:
    #                 # west_state = (x - 1, y)
    #                 north_state = (x - 1, y + 1)
    #                 if (y + 1 > self.nrows - 1) or (self.states.index(north_state) in self.obstacles):
    #                     north_state = (x, y)
    #                     north_prob = False
    #                 else:
    #                     north_state = north_state
    #             else:
    #                 # west_state = (x, y)
    #                 north_state = (x, y)
    #                 north_prob  = False
    #
    #             if x + 1 <= self.ncols - 1:
    #                 # east_state = (x + 1, y)
    #                 south_state = (x + 1, y + 1)
    #                 if (y + 1 > self.nrows - 1) or (self.states.index(south_state) in self.obstacles):
    #                     south_state = (x, y)
    #                     south_prob = False
    #                 else:
    #                     south_state = south_state
    #             else:
    #                 south_state = (x, y)
    #                 south_prob  = False
    #
    #         elif action == 'W':
    #             if y - 1 >= 0:
    #                 # south_state = (x, y - 1)
    #                 west_state = (x, y - 1)
    #                 if (self.states.index(west_state) in self.obstacles):
    #                     west_state = (x, y)
    #                     west_prob = False
    #                 else:
    #                     west_state = west_state
    #             else:
    #                 # south_state = (x, y)
    #                 west_state = (x, y)
    #                 west_prob  = False
    #
    #             if y + 1 <= self.nrows - 1:
    #                 # north_state = (x, y + 1)
    #                 east_state = (x - 1, y + 1)
    #                 if (x - 1 < 0) or (self.states.index(east_state) in self.obstacles):
    #                     east_state = (x, y)
    #                     east_prob = False
    #                 else:
    #                     east_state = east_state
    #             else:
    #                 east_state = (x, y)
    #                 east_prob  = False
    #
    #             if x - 1 >= 0:
    #                 # west_state = (x - 1, y)
    #                 north_state = (x - 1, y - 1)
    #                 if (y - 1 < 0) or (self.states.index(north_state) in self.obstacles):
    #                     north_state = (x, y)
    #                     north_prob = False
    #                 else:
    #                     north_state = north_state
    #             else:
    #                 # west_state = (x, y)
    #                 north_state = (x, y)
    #                 north_prob  = False
    #
    #             if x + 1 <= self.ncols - 1:
    #                 # east_state = (x + 1, y)
    #                 south_state = (x + 1, y - 1)
    #                 if (y - 1 < 0) or (self.states.index(south_state) in self.obstacles):
    #                     south_state = (x, y)
    #                     south_prob = False
    #                 else:
    #                     south_state = south_state
    #             else:
    #                 south_state = (x, y)
    #                 south_prob  = False
    #
    #         if action == 'N':
    #             if ((north_prob == False) and (east_prob == False) and (west_prob == False)) or (
    #                     (north_prob == True) and (east_prob == True) and (west_prob == True)):
    #                 north_probability = self.robotmdp.P(0, action, 1)
    #                 west_probability = self.robotmdp.P(0, action, 4)
    #                 south_probability = self.robotmdp.P(0, action, 3)
    #                 east_probability = self.robotmdp.P(0, action, 2)
    #             else:
    #                 if north_prob == False:
    #                     north_probability = 0
    #                 else:
    #                     north_probability = self.robotmdp.P(0, action, 1)
    #
    #                 if east_prob == False:
    #                     east_probability = 0
    #                 else:
    #                     east_probability = self.robotmdp.P(0, action, 2)
    #
    #                 if west_prob == False:
    #                     west_probability = 0
    #                 else:
    #                     west_probability = self.robotmdp.P(0, action, 4)
    #
    #                 south_probability = self.robotmdp.P(0, action, 3)
    #         elif action == 'S':
    #             if ((south_prob == False) and (east_prob == False) and (west_prob == False)) or (
    #                     (south_prob == True) and (east_prob == True) and (west_prob == True)):
    #                 north_probability = self.robotmdp.P(0, action, 1)
    #                 west_probability = self.robotmdp.P(0, action, 4)
    #                 south_probability = self.robotmdp.P(0, action, 3)
    #                 east_probability = self.robotmdp.P(0, action, 2)
    #             else:
    #                 if south_prob == False:
    #                     south_probability = 0
    #                 else:
    #                     south_probability = self.robotmdp.P(0, action, 3)
    #
    #                 if east_prob == False:
    #                     east_probability = 0
    #                 else:
    #                     east_probability = self.robotmdp.P(0, action, 2)
    #
    #                 if west_prob == False:
    #                     west_probability = 0
    #                 else:
    #                     west_probability = self.robotmdp.P(0, action, 4)
    #
    #                 north_probability = self.robotmdp.P(0, action, 1)
    #         elif action == 'E':
    #             if ((north_prob == False) and (east_prob == False) and (south_prob == False)) or (
    #                     (north_prob == True) and (east_prob == True) and (south_prob == True)):
    #                 north_probability = self.robotmdp.P(0, action, 1)
    #                 west_probability = self.robotmdp.P(0, action, 4)
    #                 south_probability = self.robotmdp.P(0, action, 3)
    #                 east_probability = self.robotmdp.P(0, action, 2)
    #             else:
    #                 if north_prob == False:
    #                     north_probability = 0
    #                 else:
    #                     north_probability = self.robotmdp.P(0, action, 1)
    #
    #                 if east_prob == False:
    #                     east_probability = 0
    #                 else:
    #                     east_probability = self.robotmdp.P(0, action, 2)
    #
    #                 if south_prob == False:
    #                     south_probability = 0
    #                 else:
    #                     south_probability = self.robotmdp.P(0, action, 3)
    #
    #                 west_probability = self.robotmdp.P(0, action, 4)
    #         else:
    #             if ((north_prob == False) and (west_prob == False) and (south_prob == False)) or (
    #                     (north_prob == True) and (west_prob == True) and (south_prob == True)):
    #                 north_probability = self.robotmdp.P(0, action, 1)
    #                 west_probability = self.robotmdp.P(0, action, 4)
    #                 south_probability = self.robotmdp.P(0, action, 3)
    #                 east_probability = self.robotmdp.P(0, action, 2)
    #             else:
    #                 if north_prob == False:
    #                     north_probability = 0
    #                 else:
    #                     north_probability = self.robotmdp.P(0, action, 1)
    #
    #                 if west_prob == False:
    #                     west_probability = 0
    #                 else:
    #                     west_probability = self.robotmdp.P(0, action, 4)
    #
    #                 if south_prob == False:
    #                     south_probability = 0
    #                 else:
    #                     south_probability = self.robotmdp.P(0, action, 3)
    #
    #                 east_probability = self.robotmdp.P(0, action, 2)
    #
    #         successors.append((north_state, north_probability))
    #         successors.append((west_state, west_probability))
    #         successors.append((south_state, south_probability))
    #         successors.append((east_state, east_probability))
    #         for (next_state, p) in successors:
    #             ns = self.states.index(next_state)
    #             prob[action][s, ns] += p
    #     return prob


class GridworldGui(Gridworld):
    def __init__(self, initial, nrows=8, ncols=8, robotmdp=MDP(), targets=[], obstacles=[], unsafe_states=[], size=100,
                 camp_sites=[], dynamic_agent_states=[]):
        super().__init__(initial, nrows, ncols, robotmdp, targets, obstacles, unsafe_states, camp_sites,
                         dynamic_agent_states)
        # compute the appropriate height and width (with room for cell borders)
        self.height = nrows * size + nrows + 1
        self.width = ncols * size + ncols + 1
        self.size = size

        # initialize pygame ( SDL extensions )
        pygame.init()
        self.gamedisplay = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Gridworld')
        self.screen = pygame.display.get_surface()
        self.surface = pygame.Surface(self.screen.get_size())
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg_rendered = False  # optimize background render

        imagebig = pygame.image.load('../../images_for_robot_examples/robot_icon.png').convert_alpha()
        self.robotimage = pygame.transform.scale(imagebig, (self.size, self.size))
        image_obs = pygame.image.load('../../images_for_robot_examples/brick.png').convert_alpha()
        self.obstacle_image = pygame.transform.scale(image_obs, (self.size, self.size))
        base_camp_A_image = pygame.image.load('../../images_for_robot_examples/base_camp_A.png').convert_alpha()
        self.base_camp_A_image = pygame.transform.scale(base_camp_A_image, (self.size, self.size))
        camp_B_image = pygame.image.load('../../images_for_robot_examples/camp_site_B.png').convert_alpha()
        self.camp_B_image = pygame.transform.scale(camp_B_image, (self.size, self.size))
        camp_C_image = pygame.image.load('../../images_for_robot_examples/camp_site_C.png').convert_alpha()
        self.camp_C_image = pygame.transform.scale(camp_C_image, (self.size, self.size))
        wave_image = pygame.image.load('../../images_for_robot_examples/wave_obstacle.png').convert_alpha()
        self.wave_image = pygame.transform.scale(wave_image, (self.size, self.size))
        drone_image = pygame.image.load('../../images_for_robot_examples/drone.png').convert_alpha()
        self.drone_image = pygame.transform.scale(drone_image, (self.size, self.size))

        # self.camp_sites = camp_sites
        self.background()
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

        self.build_templates()
        self.updategui = True  # switch to stop updating gui if you want to collect a trace quickly

        self.current = (self.mdp.init, 34)  # when start, the current state is the initial state
        self.state2circle(self.current)

        # self.drone = 32
        # self.state2circle(self.drone, drone=True)

    def build_templates(self):

        # # Note: template already in "graphics" coordinates
        template = np.array([(-1, 0), (0, 0), (1, 0), (0, 1), (1, 0), (0, -1)])
        template = self.size / 3 * template  # scale template

        v = 1.0 / np.sqrt(2)
        rot90 = np.array([(0, 1), (-1, 0)])
        rot45 = np.array([(v, -v), (v, v)])  # neg

        # align the template with the first action.
        t0 = np.dot(template, rot90)
        t0 = np.dot(t0, rot90)
        t0 = np.dot(t0, rot90)

        t1 = np.dot(t0, rot45)
        t2 = np.dot(t1, rot45)
        t3 = np.dot(t2, rot45)
        t4 = np.dot(t3, rot45)
        t5 = np.dot(t4, rot45)
        t6 = np.dot(t5, rot45)
        t7 = np.dot(t6, rot45)

        self.t = [t0, t1, t2, t3, t4, t5, t6, t7]

    def indx2coord(self, s, center=False):
        # the +1 indexing business is to ensure that the grid cells
        # have borders of width 1px
        i, j = self.coords(s)
        if center:
            return int(i * (self.size + 1) + 1 + self.size / 2), \
                   int(j * (self.size + 1) + 1 + self.size / 2)
        else:
            return int(i * (self.size + 1) + 1), int(j * (self.size + 1) + 1)

    def coord2indx(self, x, y):
        return self.rcoord((x / (self.size + 1)), (y / (self.size + 1)))

    def draw_state_labels(self):
        font = pygame.font.SysFont("FreeSans", 24)
        # for s in self.mdp.states:
        for s in range(self.nstates):
            x, y = self.indx2coord(s, False)
            txt = font.render("%d" % s, True, (0, 0, 0))
            self.surface.blit(txt, (y, x))

        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def coord2state(self, coord):
        s = self.coord2indx(coord[0], coord[1])
        return s

    def draw_state_region(self, state, region, bg=True, blit=True):
        drone = state[1]
        state = state[0]
        if bg:
            self.background()

        for s in region:
            x, y = self.indx2coord(s, False)
            coords = pygame.Rect(y, x, self.size, self.size)
            pygame.draw.rect(self.surface, ((204, 255, 204)), coords)
        x, y = self.indx2coord(state, center=True)
        # pygame.draw.circle(self.surface, (0, 0, 255), (y, x), int(self.size / 2))
        self.surface.blit(self.robotimage, (y - self.size / 4, x - self.size / 4))
        # pygame.display.update()

        x, y = self.indx2coord(drone, center=True)
        self.surface.blit(self.drone_image, (y - self.size / 4, x - self.size / 4))

        if blit:
            # self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
        return

    def state2circle(self, state, bg=True, blit=True, drone=False):
        if bg:
            self.background()

        x, y = self.indx2coord(state[0], center=True)
        x_d, y_d = self.indx2coord(state[1], center=True)
        # pygame.draw.circle(self.surface, (0, 0, 255), (y, x), int(self.size / 2))
        # self.surface.blit(self.robotimage, (y - self.size / 4, x - self.size / 4))
        self.surface.blit(self.robotimage, (y - self.size / 2, x - self.size / 2))
        self.surface.blit(self.drone_image, (y_d - self.size / 2, x_d - self.size / 2))
        # pygame.display.update()
        if blit:
            # self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

    def draw_region(self, M):

        for s in M:
            x, y = self.indx2coord(s, False)
            coords = pygame.Rect(y, x, self.size, self.size)
            pygame.draw.rect(self.bg, ((204, 255, 204)), coords)
        time.sleep(0.2)
        self.screen.blit(self.bg, (0, 0))
        pygame.display.flip()

    def draw_values(self, vals):
        """
        vals: a dict with state labels as the key
        """
        font = pygame.font.SysFont("FreeSans", 10)

        # for s in self.mdp.states:
        for s in self.states:
            x, y = self.indx2coord(s, False)
            v = vals[s]
            txt = font.render("%.1f" % v, True, (0, 0, 0))
            self.surface.blit(txt, (y, x))

        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def save(self, filename):
        pygame.image.save(self.surface, filename)

    def redraw(self):
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def follow(self, s, t, policy):
        # s is the state in the mdp and t is the state in the DRA.
        self.current = s
        action = policy[(s, t)]
        if type(action) == set:
            action = random.choice(list(action))
        self.move(action)
        time.sleep(0.1)

    def move(self, act, obs=False):
        self.current = self.mdp.sample(self.current, act)
        if self.updategui:
            self.state2circle(self.current)
        return

    def move_deter(self, next_state):
        self.current = next_state
        if self.updategui:
            self.state2circle(self.current)
        return

    def background(self):

        if self.bg_rendered:
            self.surface.blit(self.bg, (0, 0))
        else:
            self.bg.fill((0, 0, 0))
            for s in range(self.nstates):
                x, y = self.indx2coord(s, False)
                coords = pygame.Rect(y, x, self.size, self.size)
                pygame.draw.rect(self.bg, ((250, 250, 250)), coords)

            target_flag = 0
            for t in self.targets:
                x, y = self.indx2coord(t, center=True)
                coords = pygame.Rect(y - self.size / 2, x - self.size / 2, self.size, self.size)
                # pygame.draw.rect(self.bg, (0, 204, 102), coords)
                pygame.draw.rect(self.bg, (152, 251, 152), coords)
                if target_flag == 0:
                    self.bg.blit(self.camp_C_image, (y - self.size / 2, x - self.size / 2))
                    target_flag = target_flag + 1
                else:
                    self.bg.blit(self.camp_C_image, (y - self.size / 2, x - self.size / 2))

            camp_site_flag = 0
            for t in self.camp_sites:
                x, y = self.indx2coord(t, center=True)
                coords = pygame.Rect(y - self.size / 2, x - self.size / 2, self.size, self.size)
                # pygame.draw.rect(self.bg, (0, 204, 102), coords)
                pygame.draw.rect(self.bg, (152, 251, 152), coords)
                if camp_site_flag == 0:
                    self.bg.blit(self.camp_B_image, (y - self.size / 2, x - self.size / 2))
                    camp_site_flag = camp_site_flag + 1
                else:
                    self.bg.blit(self.base_camp_A_image, (y - self.size / 2, x - self.size / 2))

            """
                            # Draw Wall in black color.
             for s in self.edges:
                (x,y)=self.indx2coord(s)
                # coords = pygame.Rect(y-self.size/2, x - self.size/2, self.size, self.size)
                coords=pygame.Rect(y, x, self.size, self.size)
                pygame.draw.rect(self.bg, (192,192,192), coords) # the obstacles are in color grey

            """
            # for s in self.obstacles:
            #     (x, y) = self.indx2coord(s)
            #     coords = pygame.Rect(y, x, self.size, self.size)
            #     pygame.draw.rect(self.bg, (255, 0, 0), coords)  # the obstacles are in color red

            for s in self.obstacles:
                (x, y) = self.indx2coord(s)
                coords = pygame.Rect(y, x, self.size, self.size)
                self.bg.blit(self.obstacle_image, (y, x))
                # pygame.draw.rect(self.bg, (35, 43, 43), coords)  # the obstacles are in color black
                # self.surface.blit(self.obstacle_image, (y - self.size / 4, x - self.size / 4))
                # self.bg.blit(self.obstacle_image, (y, x))

            unsafe_flag = 0
            for s in self.unsafe_states:
                (x, y) = self.indx2coord(s)
                coords = pygame.Rect(y, x, self.size, self.size)
                pygame.draw.rect(self.bg, (255, 228, 225), coords)  # the obstacles are in color red
                # self.bg.blit(self.bomb_image, (y, x))
                if (unsafe_flag % 2) == 0:
                    self.bg.blit(self.wave_image, (y, x))
                    unsafe_flag = unsafe_flag + 1
                else:
                    self.bg.blit(self.wave_image, (y, x))
                    unsafe_flag = unsafe_flag + 1

        self.bg_rendered = True  # don't render again unless flag is set
        self.surface.blit(self.bg, (0, 0))

    def mainloop(self, dra, policy):
        """
        The robot moving in the Grid world with respect to the specification in DRA.
        """
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
        t = dra.get_transition(self.mdp.L[self.current], dra.initial_state)  # obtain the initial state of the dra

        while True:
            self.follow(self.current, t, policy)
            next_t = dra.get_transition(self.mdp.L[self.current], t)
            t = next_t
            print("The state in the DRA is {t}")
            # raw_input('Press Enter to continue ...')
            if self.current in self.walls:
                # hitting the obstacles
                print("Hitting the walls, restarting ...")
                # raw_input('Press Enter to restart ...')
                self.current = self.mdp.init  # restart the game
                print("the current state is {}".format(self.current))
                t = dra.initial_state
                self.state2circle(self.current)
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

    def mainloop2(self):
        """
        The robot moving in the Grid world with respect to the specification in DRA.
        """

        pygame.display.set_caption("hello")

        clock = pygame.time.Clock()

        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
        self.screen = pygame.display.get_surface()
        self.surface = pygame.Surface(self.screen.get_size())
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg_rendered = False  # optimize background render

        self.background()
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

        self.build_templates()
        self.updategui = True  # switch to stop updating gui if you want to collect a trace quickly

        self.current = self.mdp.init  # when start, the current state is the initial state
        self.state2circle(self.current)

        while True:
            pygame.display.update()
            clock.tick(60)
