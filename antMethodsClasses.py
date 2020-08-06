import numpy as np
import random
from antClasses import Nestmate
from operator import add
from abc import ABC, abstractmethod
import copy


# MovementCreator class ensures instatiation of a new movement object to each forager ant (via its __new__ method)
# This is required as the model receives a single Movement class which is given to each ant. Hence without the MovementCreator class, each forager would have the same movement object, so when one ant moves, they all move. With this class, each forager receives a new movement object when it is asigned the models movement object

# There are a number of different metods by which the forager can move, depending on the users parameters, the correct movement will be chosen in the if-else statment in the __new__ method
class MovementCreator:

    def __init__(self, bias=False, f_inertial_force= 0, b_inertial_force =0, inertia_weight =0, stochastic=True,
                 order_test = False, two_d = False, extreme_move = False, aver_veloc = False, step_size=False):
        self.bias =bias
        self.f_inertial_force = f_inertial_force
        self.b_inertial_force = b_inertial_force
        self.inertia_weight = inertia_weight
        self.stochastic= stochastic
        self.order_test = order_test
        self.two_d = two_d
        self.extreme_move = extreme_move
        self.aver_veloc=aver_veloc
        self.step_size = step_size

    def __new__(cls, bias, f_inertial_force, b_inertial_force, inertia_weight, stochastic, order_test, two_d,
                extreme_move, aver_veloc, step_size):
        if order_test:
            return SpacelessMovement()
    
        elif aver_veloc:
            return AverageVelocity(step_size = step_size)

        # elif extreme_move:
        #     return ExtremeMovement(bias = bias)

        elif two_d:
            return TDStochasticMovement(bias=bias, f_inertia_force=f_inertial_force, b_inertia_force=b_inertial_force,
                                        inertia_weight=inertia_weight)

        # elif stochastic is True and not order_test and not two_d:
        elif stochastic is True:
            return StochasticMovement(bias=bias, f_inertia_force=f_inertial_force, b_inertia_force=b_inertial_force,
                                      inertia_weight=inertia_weight)
        else:
            return DeterministicMovement(step_size = step_size)


# Same idea as the MovementCreator class, it ensures each ant receives a new instance of the trophallaxis object instantiated in the model class

# There are two types of trophallaxis, stochastic or deterministic, chosen based on user parameters
class TrophallaxisCreator:

    def __init__(self, stochastic=True):
        self.stochastic = stochastic

    def __new__(cls, stochastic):
        if stochastic:
            return StochasticTrophallaxis()
        else:
            return DeterministicTrophallaxis()


# Trophallaxis abstract class, lays out the methods required by all trophallaxis classes
class Trophallaxis(ABC):# abstract class used as an interface
    
    @staticmethod
    def prop_empty_space_given(agent):
        pass

    # Method determins whether donor has food to give
    @staticmethod
    def can_agent_give_food(agent):
        if agent.model.diff_test:
            # This was done to test diffusion in the model, where nestmates could only give food above a certain crop to see if diffusion allowed all ants reached said crop
            if isinstance(agent, Nestmate):
                if agent.crop_state > (agent.model.full_nest_ants/ agent.model.number_of_ants["Nestmate"]):
                    return True

        # if agent is a forager only care whether it has any food
        else:
            if agent.crop_state > 0:
                return True

    @staticmethod
    def is_neighbour_not_full(agent):
        if agent.crop_state < 1:
            return True

    @staticmethod
    def give_food_amount(agent, offered_food):
        if agent.model.diff_test:
            if isinstance(agent, Nestmate):
                return min((agent.crop_state - (agent.model.full_nest_ants/ agent.model.number_of_ants["Nestmate"])),
                            offered_food)
        else:
            return min(agent.crop_state, offered_food, (agent.interacting_neighbour.capacity -
                                                    agent.interacting_neighbour.crop_state))

    @staticmethod
    def offered_food(other_agent, variable):
        return (other_agent.capacity - other_agent.crop_state) * variable

    def trophallaxis(self, agent, model):
        gave_food = False
        proportion_given = self.prop_empty_space_given(agent)
        if agent.interacting_neighbour:
            if  isinstance(agent, Nestmate) or not isinstance(agent.movement_method, DeterministicMovement) or \
                    agent.movement_method._new_cell_flag:
                if self.can_agent_give_food(agent):
                    if self.is_neighbour_not_full(agent.interacting_neighbour):
                        offered_food = self.offered_food(agent.interacting_neighbour, proportion_given)
                        agent.food_given = self.give_food_amount(agent, offered_food)
                        agent.interacting_neighbour.crop_state += agent.food_given
                        agent.crop_state -= agent.food_given
                        gave_food = True

        agent.interaction = gave_food

    def get_sample(self):
        pass


class StochasticTrophallaxis(Trophallaxis):

    name = 'StochasticTrophallaxis'

    # Proportion of empty space given is sampled from an exponential distribution with mean defined by the user
    @staticmethod
    def prop_empty_space_given(agent):
        if agent.proportion_to_give >= 1: # maybe can make this a string, eg 'all' or 'fill'
            sample = 1
        else:
            sample = np.random.exponential(agent.proportion_to_give)
        return sample

    def trophallaxis(self, agent, model):
        super().trophallaxis(agent, model)


class DeterministicTrophallaxis(Trophallaxis):

    name = 'DeterministicTrophallaxis'

    @staticmethod
    def prop_empty_space_given(agent):
        sample = agent.proportion_to_give
        return sample

    def trophallaxis(self, agent, model):
        super().trophallaxis(agent, model)



# Movement abstract class
class Movement(ABC):

    @abstractmethod
    def move(self, agent, model):
        pass

# One dimensional movement 'abstract' class
class OneDMovement(Movement):

    # If the forager has reached the entrance it always moves into the nest
    @staticmethod
    def enter_nest(agent, model):
        if agent.pos in model.entrance or agent.position == model.entrance:
            agent.pos[0] = 1
            agent.position[0] = 1
            return True

    # If the forager reaches the end of the nest, it moves one step backwards, 1 step backwards may not be the best option as forager may still get stuck in last two cells when the colony is very full
    @staticmethod
    def move_from_edge_of_nest(agent, model):
        if int(agent.pos[0]) >= model.length or int(agent.position[0]) >= model.length:
            agent.pos[0] -= 1
            agent.position[0] -= 1
            return True

    def move(self, agent, model):
        pass


# Deterministic movement in models for which the forager only gives food the first time it enters a new cell
class DeterministicMovement(OneDMovement):

    name = 'DeterministicMovement'

    # New cell flag is true on first step in which forager enters a new cell
    def __init__(self, step_size,  new_cell_flag = False):
        self._new_cell_flag = new_cell_flag
        self.step_size = step_size

    @staticmethod
    def get_grid_position(continuous_position, step, operator):
        new_position = operator(continuous_position, step)
        new_cell = int(operator(continuous_position, step))
        return new_position, new_cell


    # If the foragers previous cell is different to its current cell, new_cell_flag is true. Forager only interacts when new_cell_flag is true, defined in trophallaxis method
    def move(self, agent, model):

        if self.enter_nest(agent, model):
            self._new_cell_flag = True
            return

        if agent.crop_state > agent.threshold:
            previous_positions = copy.copy(agent.position)
            # agent.position = [x + y for x, y in zip(agent.position, agent.step_sizes[0])]
            agent.position[0] += self.step_size[0]
            self._new_cell_flag = [np.floor(x) for x in previous_positions] != [np.floor(y) for y in agent.position]
            if agent.position[0] > model.length:
                agent.position[0] = model.length

        elif agent.crop_state < agent.threshold:
            previous_positions = copy.copy(agent.position)
            # agent.position = [x + y for x, y in zip(agent.position, agent.step_sizes[1])]
            agent.position[0] += self.step_size[1]
            if agent.position[0] < 0:
                agent.position[0] = 0
            self._new_cell_flag = [np.floor(x) for x in previous_positions] != [np.floor(y) for y in agent.position]


        if self.move_from_edge_of_nest(agent, model):
            # agent.threshold = agent.crop_state
            self._new_cell_flag = True
            pass



class StochasticMovement(OneDMovement):

    name = 'StochasticMovement'

    persistence = 0

    def __init__(self, bias, f_inertia_force= 0, b_inertia_force= 0, inertia_weight=0):
        self.movement_bias = bias
        self.weight = inertia_weight
        self.f_force = f_inertia_force
        self.b_force = b_inertia_force

    # Method that decides the foragers next move based on the biases provided. Also returns the persistence force when the forager moves with inertia
    def decide_movement(self, bias):
        coin = random.random()
        coin += (self.persistence * self.weight)
        if coin < bias[0]:
            return -1, -coin
        elif bias[0] < coin < bias[1]:
            return 0, 0
        else:
            return 1, coin

    def move(self, agent, model):

        # Foragers first step into the nest on each visit its persistence is set to the user defined forward force
        if self.enter_nest(agent, model):
            self.persistence = self.f_force
            return

        else:
            if agent.crop_state > agent.threshold:
                direc, self.persistence = self.decide_movement(self.movement_bias[0])
                # self.persistence = direc * self.persistence
                agent.position[0] += direc


            else:
                direc, self.persistence = self.decide_movement(self.movement_bias[1])
                # self.persistence = direc * self.persistence
                agent.position[0] += direc

        # If the forager reaches the edge of the nest, its persistence is set to the user defined backward force
        if self.move_from_edge_of_nest(agent, model):
            # agent.threshold = agent.crop_state
            self.persistence = self.b_force
            pass


# Movement class in which the forager remains 1 step away from exit, deciding to exit based on a function of its crop, moving forward 1 cell at each step
class SpacelessMovement(OneDMovement):


    @staticmethod
    def exit_function(x):
        if x == 0:
            return 1
        else:
            return -np.log(x)/4


    def forager_exit(self, agent):
        coin = random.random()
        prob_at_crop = self.exit_function(agent.crop_state)
        if coin <= prob_at_crop:
            agent.position = [0,0]
            return True


    def move(self, agent, model):

        if self.forager_exit(agent):
            return

        else:
            agent.position[0] += 1
            # If the forager reaches the edge of the nest, it moves 4 cells backwards
            if agent.pos[0] >= model.length or agent.position[0] >= model.length:
                agent.position[0] -= 4


# Extreme movement class, redundent
# class ExtremeMovement(OneDMovement):

#     def __init__(self, bias):
#         self.bias = bias

#     @staticmethod
#     def decide_movement(bias):
#         coin = random.random()
#         if coin < bias[0]:
#             return -1
#         else:
#             return 1

#     def move(self, agent, model):

#         if self.enter_nest(agent, model):
#             return

#         else:
#             if agent.crop_state > agent.threshold:
#                 agent.position[0] += self.decide_movement(self.bias[0])
#             else:
#                 agent.position[0] += self.decide_movement(self.bias[1])

#         if self.move_from_edge_of_nest(agent, model):
#             return


# Forager moves according to average veloicty of the biases defined by the user. Similar to deterministic movement class, however, in this case, forager interacts at every step regardless if on same cell or not
class AverageVelocity(OneDMovement):

    def __init__(self, step_size):
        self.step_size = step_size


    def move(self, agent, model):

        if self.enter_nest(agent, model):
            return

        if agent.crop_state > agent.threshold:
            agent.position[0] += self.step_size[0]
        else:
            agent.position[0] += self.step_size[1]
            if agent.position[0] < 0:
                agent.position[0] = 0

        if self.move_from_edge_of_nest(agent, model):
            return


# Two dimensional movement 'abstract' class 
class TwoDMovement(Movement):

    # Forager enters the nest at cell (1, 0) at every visit
    @staticmethod
    def enter_nest(agent, model):
        if agent.pos in model.entrance or agent.position == model.entrance:
            agent.position[0] = 1
            return True

    # Method for elementwise addition of two lists
    @staticmethod
    def add_pos(lst1, lst2):
        return list(map(add, lst1, lst2))

    # Defines which cells in Moore neighbourhood give movement forwards in 2D (when sum of x and y coordinates >0), stay in 2D (sum is 0) backwrads in 2D (when sum is less than 0)
    def move_direc(self, x):
        moves = [[x,y] for x in range(-1,2) for y in range(-1,2)]
        if x ==1:
            return [x for x in moves if sum(x) >0]
        elif x == -1:
            return [x for x in moves if sum(x) <0]
        elif x==0:
            return [x for x in moves if sum(x) ==0]


    # Ensures the forager does not go out of bounds
    def move_from_edge_of_nest(self, agent, model):

        possible_position = agent.position.copy()

        while not self.legal_move(possible_position, model):
            coord = random.choice([0,1])
            possible_position[coord] = agent.position[coord] + random.choice([-1, 1])

        agent.pos = possible_position
        agent.position = possible_position
        return True

    # legal move if position is in the model area
    @staticmethod
    def legal_move(position, model):

        # TODO: more efficient if I can use the edges, espeically for big nest
        # Can check if its out of bounds by checking coordinate vs higher/depth of nest
        # edge1 = [[0,x] for x in range(model.nest_height) if x != 0]
        # edge2 = [[model.length,x] for x in range(model.nest_height)]
        # edge3 = [[x, model.nest_height] for x in range(model.length)]
        # edge4 = [[x, -1] for x in range(model.length)]
        # edges = edge1 + edge2 + edge3 + edge4

        arena = [[x,y] for x in range(1, model.length) for y in range(model.nest_height)]
        arena = arena + [model.entrance]

        if position in arena:
            return True
        else:
            return False

    def move(self, agent, model):
        pass

class TDStochasticMovement(TwoDMovement):

    name = 'TDStochasticMovement'

    persistence = 0

    def __init__(self, bias, f_inertia_force= 0, b_inertia_force= 0, inertia_weight=0):
        self.movement_bias = bias
        self.weight = inertia_weight
        self.f_force = f_inertia_force
        self.b_force = b_inertia_force


    # Same as deterministic movement class
    def decide_movement(self, bias):
        coin = random.random()
        coin += (self.persistence * self.weight)
        if coin < bias[0]:
            return -1, -coin
        elif bias[0] < coin < bias[1]:
            return 0, 0
        else:
            return 1, coin


    def move(self, agent, model):

        if self.enter_nest(agent, model):
            self.persistence = self.f_force
            return

        # Forager chooses a random move out of the moves which correspond to her decision direction (forward, stay, backwards)
        else:
            if agent.crop_state > agent.threshold:
                direc, self.persistence = self.decide_movement(self.movement_bias[0])
                # self.persistence = direc * self.persistence
                agent.position = self.add_pos(agent.position, random.choice(self.move_direc(direc)))


            else:
                direc, self.persistence = self.decide_movement(self.movement_bias[1])
                # self.persistence = direc * self.persistence
                agent.position = self.add_pos(agent.position, random.choice(self.move_direc(direc)))


        if self.move_from_edge_of_nest(agent, model):
            # agent.threshold = agent.crop_state
            self.persistence = self.b_force
            pass



