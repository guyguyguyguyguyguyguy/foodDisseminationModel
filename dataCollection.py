from FoodAgent import Forager, Nestmate
import numpy as np

# Methods in this file are used for data collection in the model

# Colony state, taken to be the average crop of all non-forager ants in the model
def get_colony_state(model):
    number_of_ants = len([x for x in model.agents if not isinstance(x, Forager)])
    total_crop = sum([x.crop_state for x in model.agents if not isinstance(x, Forager)])
    colony_crop = total_crop/number_of_ants
    return colony_crop


def get_empty_colony_state(model):
    return 1 - get_colony_state(model)


def model_step(model):
    return model.steps


def get_foragers_at_exit(model):
    foragers_at_exit = [x for x in model.agents if isinstance(x, Forager) and x.position ==[0, 0]and x.lag_counter >= 1]

    return foragers_at_exit


def get_exiting_crop(model):
    return [j.exiting_crop for j in get_foragers_at_exit(model)]


def get_steps_to_exit(model):
    return [j.trip_length for j in get_foragers_at_exit(model)]


def get_interacting_forager(model):
    interacting_foragers = [x for x in model.agents if isinstance(x, Forager) and x.interaction is True]

    return interacting_foragers


def get_interacting_forager_crop(model):
    interacting_foragers_crop = [x.crop_state for x in get_interacting_forager(model)]
    return interacting_foragers_crop


def get_recipients(model):
    interacting_forager_positions = [x.position for x in get_interacting_forager(model)]
    recipients = [y for y in model.agents if isinstance(y, Nestmate) and y.position in
                      interacting_forager_positions]

    return recipients


def get_interaction_volume(model):
    volume_given_by_interacting_forager = [x.trophallaxis_method.food_given for x in get_interacting_forager(
        model)]

    return volume_given_by_interacting_forager

