from setup_and_solvers.markov_decision_process import *

import time

import pygame
# import pygame.locals as pgl

# import random as pr
import random


class Gridworld():
    def __init__(self, current=0, nrows=8, ncols=8, robotmdp=MDP(), targets=[], obstacles=[], unsafe_states=[],
                 initial_dist=dict([])):
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
        self.states = [(x, y) for x in range(0, ncols) for y in range(0, nrows)]

        # # Trying to remove the obstacles from the list of states.
        # self.states_without_obstacles = list(set(range(self.nstates)) - set(self.obstacles))

        prob = {a: np.zeros((self.nstates, self.nstates)) for a in self.actlist}
        for s in self.states:
            for a in self.actlist:
                prob = self.getProbs(s, a, prob)
        self.mdp = MDP(current, self.actlist, range(self.nstates))  # TODO: Change this to suit the new MDP!
        self.mdp.prob = prob
        self.mdp.goal_states = set(targets)
        self.mdp.initial_distribution = initial_dist

    def coords(self, s):
        return self.states[s]  # the coordinate for state s.

    def rcoord(self, x, y):
        return self.states.index((x, y))


    # The following is the normal NSEW dynamics - working.
    def getProbs(self, state, action, prob):
        successors = []
        (x, y) = state
        s = self.states.index(state)
        if (s in self.obstacles) or (s in self.targets) or (s in self.unsafe_states):
            prob[action][s, s] = 1
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
                ns = self.states.index(next_state)
                prob[action][s, ns] += p
        return prob


class GridworldGui(Gridworld):
    def __init__(self, initial, nrows=8, ncols=8, robotmdp=MDP(), targets=[], obstacles=[], unsafe_states=[],
                 initial_dist=dict([]), size=100):
        super().__init__(initial, nrows, ncols, robotmdp, targets, obstacles, unsafe_states, initial_dist)
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

        imagebig = pygame.image.load('images_for_example/mario.png').convert_alpha()
        # imagebig = pygame.image.load(
        #     'C:/Users/sudupa/Desktop/Sensor_deception_with_hidden_sensor/sensor_deception_hidden_sensor_setup/Examples/figures_for_Mario_example/mario.png').convert_alpha()

        self.robotimage = pygame.transform.scale(imagebig, (self.size, self.size))
        image_obs = pygame.image.load('images_for_example/brick.png').convert_alpha()
        # image_obs = pygame.image.load(
        #     'C:/Users/sudupa/Desktop/Sensor_deception_with_hidden_sensor/sensor_deception_hidden_sensor_setup/Examples/figures_for_Mario_example/brick.png').convert_alpha()

        self.obstacle_image = pygame.transform.scale(image_obs, (self.size, self.size))
        image_princess = pygame.image.load('images_for_example/princess.png').convert_alpha()
        # image_princess = pygame.image.load(
        #     'C:/Users/sudupa/Desktop/Sensor_deception_with_hidden_sensor/sensor_deception_hidden_sensor_setup/Examples/figures_for_Mario_example/princess.png').convert_alpha()

        self.princess_image = pygame.transform.scale(image_princess, (self.size, self.size))
        image_luigi = pygame.image.load('images_for_example/luigi.png').convert_alpha()
        # image_luigi = pygame.image.load(
        #     'C:/Users/sudupa/Desktop/Sensor_deception_with_hidden_sensor/sensor_deception_hidden_sensor_setup/Examples/figures_for_Mario_example/luigi.png').convert_alpha()

        self.luigi_image = pygame.transform.scale(image_luigi, (self.size, self.size))
        image_bomb = pygame.image.load('images_for_example/bo-omb.png').convert_alpha()
        # image_bomb = pygame.image.load(
        #     'C:/Users/sudupa/Desktop/Sensor_deception_with_hidden_sensor/sensor_deception_hidden_sensor_setup/Examples/figures_for_Mario_example/bo-omb.png').convert_alpha()

        self.bomb_image = pygame.transform.scale(image_bomb, (self.size, self.size))
        image_plant = pygame.image.load('images_for_example/plant_land_piranha.png').convert_alpha()
        # image_plant = pygame.image.load(
        #     'C:/Users/sudupa/Desktop/Sensor_deception_with_hidden_sensor/sensor_deception_hidden_sensor_setup/Examples/figures_for_Mario_example/plant_land_piranha.png').convert_alpha()

        self.plant_image = pygame.transform.scale(image_plant, (self.size, self.size))

        toad_image = pygame.image.load('images_for_example/toad.png').convert_alpha()
        # image_plant = pygame.image.load(
        #     'C:/Users/sudupa/Desktop/Sensor_deception_with_hidden_sensor/sensor_deception_hidden_sensor_setup/Examples/figures_for_Mario_example/plant_land_piranha.png').convert_alpha()

        self.toad_image = pygame.transform.scale(toad_image, (self.size, self.size))

        self.background()
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

        self.build_templates()
        self.updategui = True  # switch to stop updating gui if you want to collect a trace quickly

        # self.current = self.mdp.init  # when start, the current state is the initial state
        self.current = random.choice(list(initial))
        self.state2circle(self.current)

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
        for s in self.mdp.states:
            x, y = self.indx2coord(s, False)
            txt = font.render("%d" % s, True, (0, 0, 0))
            self.surface.blit(txt, (y, x))

        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def coord2state(self, coord):
        s = self.coord2indx(coord[0], coord[1])
        return s

    def draw_state_region(self, state, region, bg=True, blit=True):
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
        if blit:
            # self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
        return

    def state2circle(self, state, bg=True, blit=True):
        if bg:
            self.background()

        x, y = self.indx2coord(state, center=True)
        # pygame.draw.circle(self.surface, (0, 0, 255), (y, x), int(self.size / 2))
        # self.surface.blit(self.robotimage, (y - self.size / 4, x - self.size / 4))
        self.surface.blit(self.robotimage, (y - self.size / 2, x - self.size / 2))
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

        for s in self.mdp.states:
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
                    self.bg.blit(self.princess_image, (y - self.size / 2, x - self.size / 2))
                    target_flag = target_flag + 1
                elif target_flag == 1:
                    self.bg.blit(self.luigi_image, (y - self.size / 2, x - self.size / 2))
                    target_flag += 1
                else:
                    self.bg.blit(self.toad_image, (y - self.size / 2, x - self.size / 2))

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
                    self.bg.blit(self.plant_image, (y, x))
                    unsafe_flag = unsafe_flag + 1
                else:
                    self.bg.blit(self.bomb_image, (y, x))
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
