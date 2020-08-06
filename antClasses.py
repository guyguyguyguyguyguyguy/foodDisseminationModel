import copy
import random
from mesa import Agent
import itertools
from Inspect import Inspectable
import uuid
import helper_functions as hf
import numpy as np

# 'Abstract' ant class
class Ant(Agent, Inspectable):
    _id =0
    agent_step = 0
    food_given = 0
    count_trip_length = 0
    interaction = False
    interacting_neighbour = None
    trip_length = None
    pos = None

    # Ant id is incremented by one on each instantiation of a class that inherits from Ant
    def __init__(self, model, position):
        super().__init__(Ant._id, model)
        Ant._id += 1
        self._position = position

    @property
    def position(self):
        return self._position

    # pos required by mesa module, however, it is not used anymore
    @position.setter
    def position(self, value):
        self._position = value
        self.pos = [int(x) for x in self.position]

    def get_interaction_neighbour(self):
        pass

    # Add agent to model agents and scheduling for abm
    def add(self):
        if self.model.inert_nestants and isinstance(self, Nestmate):
            self.model.agents.append(self)
        else:
            self.model.agents.append(self)
            self.model.schedule.add(self)

    # Redundent
    def place(self):
        self.model.grid.place_agent(self, self.position)

    def step(self):
        pass


class Forager(Ant):
    crop_state = 1
    exiting_crop = 0
    trip = 0
    lag_counter= 0
    at_leading_edge = False
    crop_at_le = None
    crop_below_thresh = False
    previous_crop = 0

    def __init__(self, model, position, motion_threshold, step_sizes, trophallaxis_method, movement_method,
                 forager_proportion_to_give):
        super().__init__(model, position)

        self.threshold = motion_threshold
        self.step_sizes = step_sizes
        self.trophallaxis_method = trophallaxis_method
        self.movement_method = movement_method
        self.proportion_to_give = forager_proportion_to_give
        self.interaction_rate = self.model.forager_interaction_rate


    def trophallaxis(self):
        self.trophallaxis_method.trophallaxis(self, self.model)

    def move(self):
        self.movement_method.move(self, self.model)


    # Choose one other ant on same cell for forager to interact with
    def get_interaction_neighbour(self):
        # interacting_neighbour = list(filter((lambda x: x.position == agent.pos and x.unique_id !=
        #                                                agent.unique_id),neighbours))

        interacting_neighbours = [x for x in self.model.agents if x.position == [np.floor(coor) for coor in self.position] and x.unique_id !=
        self.unique_id and isinstance(x, Nestmate)]


        if interacting_neighbours:
            interacting_neighbour = random.choice(interacting_neighbours)
            self.interacting_neighbour = interacting_neighbour

        else:
            self.interacting_neighbour = False


    def crop_drop(self):
        if self.crop_state <= self.threshold and self.previous_crop > self.threshold:
            self.crop_below_thresh = True
        else:
            self.crop_below_thresh = False

    def step(self):
        self.previous_crop = copy.copy(self.crop_state)
        if self.model.diff_test:
            self.agent_step = self.model.model_step
            self.move()
            pass

        else:
            self.agent_step = self.model.model_step

            if self.position == [0, 0]:
                self.trip_length = self.count_trip_length
                # print('waiting %s, position %s' %(self.lag_counter, self.position))
                if self.agent_step == 1 or self.lag_counter >= self.model.forager_lag:
                    self.crop_state = 1
                    self.count_trip_length = 0
                    self.trip += 1
                    self.move()
                    self.get_interaction_neighbour()
                    if (not self.at_leading_edge) and self.interacting_neighbour and (
                            self.interacting_neighbour.crop_state < 0.9):
                        self.crop_at_le = self.crop_state
                        self.at_leading_edge = True
                    else:
                        self.crop_at_le = None
                    if hf.decision_making(self.interaction_rate):
                        self.trophallaxis()
                    self.lag_counter = 1
                    self.exiting_crop = None
                    # print('Done waiting')
                else:
                    self.lag_counter += 1



            elif self.position != [0, 0]:
                self.move()
                self.get_interaction_neighbour()
                if (not self.at_leading_edge) and self.interacting_neighbour and  (
                        self.interacting_neighbour.crop_state < 0.85):
                    self.crop_at_le = self.crop_state
                    self.at_leading_edge = True
                else:
                    self.crop_at_le = None
                if hf.decision_making(self.interaction_rate):
                    self.trophallaxis()
                self.count_trip_length += 1
                if self.position == [0,0]:
                    self.exiting_crop = self.crop_state
                    self.trip_length = self.count_trip_length
                    self.at_leading_edge =False

                else:
                    self.exiting_crop = None
                # self.trip_length = None

        self.crop_drop()


class Nestmate(Ant):
    crop_state = 0
    interaction_rate = None
    position_selector = None

    def __init__(self, model, position, nestmate_capacity, trophallaxis_method, nestmate_proportion_to_give):
        super().__init__(model, position)

        self.capacity = nestmate_capacity
        self.trophallaxis_method = trophallaxis_method
        self.proportion_to_give = nestmate_proportion_to_give

        if self.model.diff_test:
            if self.unique_id in range(1, self.model.full_nest_ants+1):
                self.crop_state = 1


    def get_interaction_neighbour(self):
        interacting_neighbours = [x for x in self.model.agents if
                                  abs(sum(x.position) - sum(self.position)) <=1 and x is not self and isinstance(x, Nestmate)]
        interacting_neighbour = random.choice(interacting_neighbours)

        if interacting_neighbour:
            self.interacting_neighbour = interacting_neighbour

        else:
            self.interacting_neighbour = False

    def get_interaction_rate(self):
        a_rate = self.model.propagate_food
        """
        Something, maybe colony state dependent, maybe space or time dependent
        """
        self.interaction_rate = a_rate

    def trophallaxis(self):
        if hf.decision_making(self.interaction_rate):
            self.trophallaxis_method.trophallaxis(self, self.model)


    def step(self):
        self.agent_step = self.model.model_step
        if self.model.propagate_food > 0 and self.crop_state > 0:
            self.get_interaction_neighbour()
            self.get_interaction_rate()
            self.trophallaxis()
        else:
            pass
