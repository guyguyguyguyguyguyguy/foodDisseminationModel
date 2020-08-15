from mainClass import Main
import sys
import ast
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import pylab as pl
import statistics
import matplotlib.colorbar as cbar
from os import sep
from scipy.signal import savgol_filter
from scipy.stats import linregress
import matplotlib.animation as animation
from antDeploymentClass import *


count = (i for i in range(2, 100))


def make_plot_of_nest_layout():
	pass


# Running function, checks whether model has been initalised through gui or directly, then sets parameter values and initalises running of model by instantiating main class
def f():
	if len(sys.argv) > 1:
		steps = int(sys.argv[next(count)])
		repeats = int(sys.argv[next(count)])
		threshold = float(sys.argv[next(count)])
		bias_above = ast.literal_eval(sys.argv[next(count)])
		bias_below = ast.literal_eval(sys.argv[next(count)])
		forward_inertia = float(sys.argv[next(count)])
		backward_inertia = float(sys.argv[next(count)])
		vel_above = float(sys.argv[next(count)])
		vel_below = float(sys.argv[next(count)])
		verbose = bool(int(sys.argv[next(count)]))
		give_at_every_step = bool(int(sys.argv[next(count)]))
		shuffle_at_exit = bool(int(sys.argv[next(count)]))
		select_ant_deployment = bool(int(sys.argv[next(count)]))
		save = sys.argv[next(count)]
		file_name = sys.argv[next(count)]
		nestmates_vel = int(sys.argv[next(count)])
		nest_depth = int(sys.argv[next(count)])
		nest_height = int(sys.argv[next(count)])
		troph = sys.argv[next(count)]
		move = sys.argv[next(count)]
		lag_len = int(sys.argv[next(count)])
		nestmate_bias = bool(int(sys.argv[next(count)]))
		nestmate_int_rate = float(sys.argv[next(count)])
		parralelise = int(sys.argv[next(count)])

	else:
		steps = 2000
		repeats = 1
		threshold = 0.3
		bias_above = [0.3, 0.65, 1]
		bias_below = [0.5, 0.75, 1]
		forward_inertia = 0
		backward_inertia = 0
		vel_above = 0.05
		vel_below = -0.25
		verbose = True
		give_at_every_step = False
		shuffle_at_exit = True
		select_ant_deployment = True
		save = "/home/gui/Downloads/guitest"
		file_name = "ahhh"
		nestmates_vel = 0
		nest_depth = 10
		nest_height = 10
		troph = "Stochastic"
		move = "Stochastic"
		lag_len = 1
		nestmate_bias = False
		nestmate_int_rate = 0
		parralelise = 1

	inertia = None
	two_d = None
	inactive_nestmate = None

	step_sizes = [vel_above, vel_below]

	if troph == "Stochastic":
		stoch_trop = True
	else:
		stoch_trop = False

	if forward_inertia > 0 and backward_inertia > 0:
		inertia = 1
	else:
		inertia = 0

	if nest_height > 1:
		two_d = True

	if move == 'Stochastic':
		stoch_mov = True
	elif move == 'Deterministic':
		stoch_mov = False
	else:
		return Exception("Movement not selected")


	if nestmates_vel > 0 or nestmate_int_rate > 0 and nestmate_bias > 0:
		inactive_nestmate = False
	else:
		inactive_nestmate = True



	# Todo: make plot of the nest layout
	if select_ant_deployment:
		selection = AntSelection(nest_depth, nest_height)
	else:
		selection = AntSelection(nest_depth, nest_height, full_nest = True)
	# print('here')
	selection.main()
	deployment = {'Forager': [[0,0]], 'Nestmate': selection.selected_ants}
	number_of_ants = {'Forager': 1, 'Nestmate': len(deployment['Nestmate'])}

	# print("Everything is working, yay, now to celebrate!")
	# print('save is {}'.format(save))
	# print('steps is {}'.format(steps))
	# print("nestmates are inhert: {}".format(inactive_nestmate))
	# print("nest is shuffled: {}".format(shuffle_at_exit))
	# print("step sizes: {}".format(step_sizes))
	a = Main(steps=steps, inertia=inertia, f_inertial_force=forward_inertia,
			 b_inertial_force=backward_inertia, bias_above=bias_above,
			 bias_below=bias_below, step_sizes=step_sizes,
			 repeat_no=repeats, verbose=verbose, save=save,
			 space_test=shuffle_at_exit,
			 homogenise=False, nestmate_movement=nestmates_vel,
			 nest_depth=nest_depth,
			 nest_height=nest_height, stoch_troph=stoch_trop, stoch_mov = stoch_mov,
			 lag_length=lag_len,
			 aver_veloc = give_at_every_step,
			 nestmate_bias=nestmate_bias,
			 propagate_food_rate=nestmate_int_rate, two_d=two_d,
			 inert_nestants=inactive_nestmate, threshold=threshold, deployment = deployment, number_of_ants=number_of_ants)

	# print('after running, pre data')

	step = a.all_step_data_average
	forag = a.all_forager_data
	visit = a.all_visit_data
	inter = a.all_interaction_data

	# print('after instantiation')
	if save:
		# print('saving here')
		step.to_csv(save+sep +'{}_step_data.csv'.format(file_name))
		forag.to_csv(save +sep+'{}_forag_data.csv'.format(file_name))
		visit.to_csv(save +sep+'{}_visit_data.csv'.format(file_name))
		inter.to_csv(save +sep+'{}_interaction_data.csv'.format(file_name))



# print('running')
f()
