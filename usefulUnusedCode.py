# from Plotting_functions import *
# from helper_functions import *
# import pandas as pd
# import glob
# import re
# import os
# names = figure_name()
# numbers = figure_number()



# #  PLOTTING

# data = [pd.read_csv("Data/StochasticMovement_StochasticTrophallaxis_03._0.15_True.csv")]
# #
# data=data.dropna(subset=['exiting crop'])
#
# data = data.drop_duplicates(subset=['exiting crop'], keep='last')
# x = data['index1']
# y = data['exiting crop']
# plt.scatter(x,y, s=6, color='blue')
# plt.xlim(0, 18000)
# plt.xlabel('steps')
# plt.ylabel('forager exiting crop')
# plt.title('exiting crop state as a function of steps')
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# import statistics as st
# from scipy.optimize import curve_fit
# from helper_functions import *
# plt.style.use('ggplot')


# def fitting_a_curve(func, x, y, colour=None, label=None, plotting=True):
#     popt, pcov = curve_fit(func, x, y)
#     x_fit = np.asarray(x)
#     # x_fit = np.linspace(0,1)
#     y = func(x_fit, *popt)
#     if plotting:
#         if colour:
#             plt.plot(x_fit, y, 'r--', color = colour, label=label, zorder=1)  # label=func.__name_
#         else:
#             plt.plot(x_fit, y, 'r--', label=label, zorder=1)  # label=func.__name_
#         return popt

#     else:
#         return popt


# def simple_two_variable_plot(data, parameter):

#     x1 = parameter[0]
#     y1 = parameter[1]

#     data = data[data['trip length'] <= 1500]

#     x = data[x1]
#     y = data[y1]

#     plt.ylabel(y1)
#     plt.xlabel(x1)

#     plt.scatter(x, y, s=0.7, color='blue')


# def plot_first_variable_average(data, parameters, legend= None, stdev = True):
#     values_all_runs = data[parameters[0]]
#     values_mean = []
#     values_std = []
#     for i in range(max(data.index1)):
#         state_at_step = values_all_runs[values_all_runs.index == i].values.tolist()
#         values_mean.extend(st.mean(state_at_step))
#         values_std.extend(st.stdev(state_at_step))

#     x = parameters[1]

#     if stdev:
#         plt.errorbar(x, values_mean, values_std, values_std, ecolor="burlywood", elinewidth=0.05, capsize=0.2,
#                  zorder=0)
#     else:
#         plt.scatter(x, values_mean, s=0.7, color ='blue')

#     plt.legend(legend, markerscale=5)


# def plot_each_run_seperatly(data, title, parameters, number_of_repeats, lst=None):
#     fig = plt.figure()
#     axes = fig.subplots(nrows = number_of_repeats//2, ncols=2)
#     axes = axes.flatten()
#     for n, run_data in enumerate(data):
#         y = run_data[parameters[0]]
#         y = run_data
#         x = parameters[1]
#         ax = fig.add_subplot(axes[n])
#         ax.scatter(x, y, s=2)
#         ax.set_xlabel("steps")
#         ax.set_ylabel(parameters[0])
#         ax.set_title(lst[n])
#     fig.suptitle(title)
#     plt.tight_layout()


# def final_value_average(data, parameters, stdev = True):
#     final_state = []
#     data = data[parameters[0]]
#     for i in range(max(data.step)):
#         values = data[data.index == i].values.tolist()
#         final_state.append(values[-1])

#     final_state_mean = st.mean(final_state)
#     final_state_std = st.stdev(final_state)

#     x = parameters[1]

#     if stdev:
#         plt.errorbar(x, final_state_mean, final_state_std, final_state_std, ecolor="burlywood", elinewidth=0.05,
#                      capsize=0.2, zorder=0)


#     plt.scatter(x, final_state_mean, color="green", s=10)


# def trip_frequency_vs_empty_colony_state(data, label=False):

#     x1 = 'empty colony state'
#     y1 = 'trip length'


#     # model_data = data.round({x1: 1})
#     model_data = data
#     model_data[x1] = model_data[x1].apply(lambda j: round(j*10)/10)
#     model_data = model_data[model_data[x1] <= 0.9]
#     model_data = model_data.groupby(x1).mean().reset_index()

#     def division_by_zero(a, b):
#         return 0 if a == 0 or b == 0 else a / b

#     y = model_data[y1].apply(lambda x: division_by_zero(1,x))
#     x = model_data[x1]

#     plt.ylabel("trip frequency")
#     plt.xlabel(x1)


#     if label:
#         plt.plot(x, y, 'o-', label=label, c='blue')
#         # plt.title(title)
#         # plt.legend()
#     else:
#         plt.plot(x, y, 'o-')

#         plt.title('trip frequency vs empty colony state')


#     return x, y


# def different_nestmate_crops(data, num=20, plotting=True):
#     ks=[]
#     x0s =[]
#     poss = []

#     data['id'] = data['id'].astype(int)
#     n = max(data['id'] + 1) / num
#     colors = plt.cm.get_cmap('winter')
#     color = iter(colors(np.linspace(0, 1, n)))

#     for i in range(2, max(data['id'])+1, num):
#         ant = data[data['id'] == i]

#         x = data['index1'].unique()
#         y = ant['crop']
#         if plotting:
#             plt.scatter(x, y, label = i, s=1.5, color = next(color), zorder=0)

#             k,x0 = fitting_a_curve(logistic_fun, x, y, colour = 'red')

#             ks.append(k)
#             x0s.append(x0)
#             poss.append(i)
#             plt.legend(loc='upper right')

#         else:
#             k, x0 = fitting_a_curve(logistic_fun, x, y, colour='red', plotting=False)

#             ks.append(k)
#             x0s.append(x0)
#             poss.append(i)

#     return ks, x0s, poss


# def half_filled_nestmtae_position(data):

#     data['index1'] = data['index1'].astype(int)

#     past_mid = data.loc[(data.crop > 0.5).groupby(data.id).idxmax]
#     # past_mid = past_mid[10:]
#     x = past_mid['colony state']
#     y = past_mid['id']

#     return x,y


# def state_of_colony_at_diff_times(data, colour=None):
#     for i in range(1000, len(data.index1.unique()), 800):

#         colony_at_step = data[data['index1'] == i]

#         x = np.linspace(0, len(colony_at_step.id), len(colony_at_step.index1))

#         plt.plot(x, colony_at_step.crop, '-o', markersize=3, color=colour, label=i)




# def leading_edge():
#     for no, file in enumerate(open_csv_files()):
#         data = pd.read_csv(file)
#         data.name = os.path.splitext(os.path.split(file[0:])[1])[0]

#         data['index1'].iloc[41:] +=1

#         f_name = next(names)
#         f_name = next(numbers)
#         state_of_colony_at_diff_times(data)
#         plt.title('crop state of all ants at different time points \n to look at edge ' + data.name)
#         plt.ylabel('crop')
#         plt.xlabel('position')
#         plt.legend(loc='upper left')
#         plt.show()
#     input()

# # leading_edge()



# def position_vs_half_filled():
#     for no, file in enumerate(open_csv_files()):
#         data = pd.read_csv(file)
#         data.name = os.path.splitext(os.path.split(file[0:])[1])[0]

#         data['index1'].iloc[41:] += 1

#         f_name = next(names)
#         f_name = next(numbers)
#         x, y = half_filled_nestmtae_position(data)
#         a,b = fitting_a_curve(linear_fun, x[10:], y[10:], colour='blue')
#         label = 'fitted line: n = {} + {}x'.format(round(a,3), round(b,3))
#         plt.scatter(x, y, s=20, label=label)
#         plt.title('position of half filled ant vs colony state \n' + data.name)
#         plt.xlabel('colony state')
#         plt.ylabel('half filled nestmate position')
#         plt.xlim(-0.01, 1.05)
#         plt.ylim(-1, len(y) + 2)
#         plt.legend()


#         plt.show()
#     input()

# # position_vs_half_filled()


# def exponenital_fit_vs_threshold():
#     xs = []
#     ys = []
#     f_name = next(names)
#     f_name = next(numbers)
#     for no, file in enumerate(open_csv_files()):
#         data = pd.read_csv(file)
#         data.name = os.path.splitext(os.path.split(file[0:])[1])[0]

#         data['index1'].iloc[41:] += 1

#         if '0.' in data.name and 'Det' not in data.name:
#             ks, x0s, poss = different_nestmate_crops(data, 1, plotting=False)

#             a, b = fitting_a_curve(inverse, poss[:-11], ks[:-11], colour='black', plotting=False)
#             a = round(a, 3)
#             b = str(round(b, 3))
#             threshold = data.name.find('0')
#             xs.append(float(data.name[threshold:threshold+3]))
#             ys.append(a)


#     plt.scatter(xs, ys, s=10)
#     plt.title('value of, a, that fits k when fitting logistic to nestmate crop \n dynamics using b + '
#               '1/(ax)')
#     plt.show()

# # exponenital_fit_vs_threshold()



# def fit_nestmate_dynamics():
#     for no, file in enumerate(open_csv_files()):
#         data = pd.read_csv(file)
#         data.name = os.path.splitext(os.path.split(file[0:])[1])[0]

#         data['index1'].iloc[41:] += 1
#         f_name = next(names)
#         f_name = next(numbers)


#         ks, x0s, poss = different_nestmate_crops(data, 1)
#         plt.title("Nestmates, at different positions, crop as a function of time \n"
#                   "fit with logistic function \n" + data.name)

#         f_name = next(names)
#         f_name = next(numbers)
#         a, b = fitting_a_curve(inverse, poss[:-11], ks[:-11], colour='black')
#         a = 'x' + str(round(a, 3))
#         b = str(round(b,3))

#         label = 'inverse, n = {0} + 1/({1}x)'.format(b, a)

#         plt.scatter(poss[:-11], ks[:-11], s=11, color='green', zorder=2, label=label)
#         plt.title('k of logistic fit to nestmate crop vs nestmate position \n'
#                   'fit with exponent \n' + data.name)
#         plt.legend(loc='upper right')

#         f_name = next(names)
#         f_name = next(numbers)
#         a,b =fitting_a_curve(powerla, poss[:-11], x0s[:-11], colour='black')
#         b = str(round(b,3))

#         label = r'powerlaw, $n$ = {} $\times$ x$^{{{}}}$'.format(b, a)
#         plt.scatter(poss[:-11], x0s[:-11], s=11, color = 'green', zorder= 2, label = label)
#         plt.title('x0 of logistic fit to nestmate crop vs nestmate position \n'
#                   'fit with power law\n' + data.name)
#         plt.legend(loc= 'upper left')

#         plt.show()
#     input()

# # fit_nestmate_dynamics()


# def frequency_plot(split=False):
#     xs=[]
#     ys=[]

#     if not split:

#         for no, file in enumerate(open_csv_files(data_directory="Visit_data/Interaction_rate")):
#             data = pd.read_csv(file)
#             data.name = os.path.splitext(os.path.split(file[0:])[1])[0]

#             f_name = next(names)
#             f_name = next(numbers)
#             x, y = trip_frequency_vs_empty_colony_state(data, title= 'trip frequency \n' + data.name)

#             a, b = fitting_a_curve(linear_fun, x, y, colour='black')
#             a = 'x' + str(round(a,3))
#             b = str(round(b, 3))

#             label = r'linear, $n$ = {} x{}'.format(a, b)
#             plt.annotate(label, xy=(0.05, 0.95), xycoords='axes fraction')

#             f_name = next(names)
#             f_name = next(numbers)
#             data = pd.read_csv(file)
#             data.name = os.path.splitext(os.path.split(file[0:])[1])[0]
#             simple_two_variable_plot(data, ['empty colony state', 'trip length'])
#             plt.title('trip length vs empty colony state \n' + data.name)
#         plt.show()
#         input()

#     else:
#         label = []
#         dataa = []
#         for no, file in enumerate(open_csv_files(data_directory="Visit_data")):
#             data = pd.read_csv(file)
#             data.name = os.path.splitext(os.path.split(file[0:])[1])[0]

#             # if "lag" in data.name and "move_False" in data.name and "Det" in data.name:
#             #     x, y = trip_frequency_vs_empty_colony_state(data, ['empty colony state', 'trip length'],
#             #                                                 remove_small_trips=False)

#                 #
#             la = re.search("[g]\_(\w+)(?!\w)", data.name)
#             if la:
#                 label.append(la.group(1))
#                 dataa.append(data)

#         for lab in label:
#             lavs = []
#             for dat in dataa:
#                 if lab ==  re.search("[g]\_(\w+)(?!\w)", dat.name).group(1):
#                     trip_frequency_vs_empty_colony_state(dat, title=lab)
#                     if "Det" in dat.name:
#                         lavs.append('det')
#                     else:
#                         lavs.append("stoc")

#             plt.legend(lavs)

#             plt.show()
#         input()

# # frequency_plot(split=False)


# def forager_below_thresh():
#     data = pd.read_csv("Data/forager_StochasticMovement_StochasticTrophallaxis_0.9_0.15.csv")
#     f_name = next(names)
#     f_name = next(numbers)

#     data['index1'].iloc[1:] +=1
#     below_thresh = data.loc[(data.crop < 0.3).groupby(data.trip).idxmax]
#     below_thresh = below_thresh[below_thresh['crop'] != 1]

#     y = below_thresh['colony state']
#     x = below_thresh['index1']
#     plt.scatter(x,y, s= 3)
#     plt.xlabel('Step')
#     plt.ylabel('Colony state')
#     plt.title('Colony state and time at which forager crop dropped \n below threshold')

#     f_name = next(names)
#     f_name = next(numbers)
#     x = below_thresh['position']
#     plt.scatter(x, y, s=4)
#     plt.xlabel('position')
#     plt.ylabel('Colony state')
#     plt.title('Colony state and position at which forager crop dropped \n below threshold')
#     plt.show()
#     input()

# # forager_below_thresh()




# # # crop of forager vs colony state, with exiting points labeled
# def cropvscolonyexit(data):
#     f_name = next(names)
#     f_name = next(numbers)

#     # data = data[data.index < 20000]

#     c = data['position']
#     c = [x==0 for x in c]

#     y = data['crop']
#     x = data['colony state']

#     plt.scatter(x,y, c=c, s=10, cmap='Paired')

# Points where forager seems to exit but doesnt start again is because its plotting 15 runs

# data = pd.read_csv(
#     'Data/Forager_data/forager_StochasticMovement_StochasticTrophallaxis_0.3_0'
#     '.15_homogenise_False_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.1_w10.csv')
# cropvscolonyexit(data)
#
# data = pd.read_csv('Data/Forager_data/forager_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_False_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.15.csv')
# cropvscolonyexit(data)
#
# data = pd.read_csv('Data/Forager_data/forager_StochasticMovement_StochasticTrophallaxis_0.3_0'
#                    '.15_homogenise_False_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.175.csv')
# cropvscolonyexit(data)

# data = pd.read_csv('Data/Forager_data/forager_StochasticMovement_StochasticTrophallaxis_0.3_0'
#                    '.15_homogenise_False_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.1.csv')
# cropvscolonyexit(data)

# data = pd.read_csv('Data/oder_forag_data.csv')
# data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_forag_data.csv')
# cropvscolonyexit(data)
# plt.title('Forager crop vs colony state \n (brown dots indicate forager has exited)')
# plt.ylabel('Foragers crop')
# plt.xlabel('Colony state')
# plt.show()

# data = pd.read_csv('Data/Forager_data/forager_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.175.csv')
# cropvscolonyexit(data)
#
# data = pd.read_csv('Data/Forager_data/forager_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.csv')
# data= data[data['colony state'] < 0.95]
# cropvscolonyexit(data)
#
# plt.show()


# data = pd.read_csv('Data/Visit_data/visit_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.csv')
# # trip_frequency_vs_empty_colony_state(data, title='0')
#
# data = pd.read_csv('Data/Visit_data/visit_StochasticMovement_StochasticTrophallaxis_0.3_0'
#                    '.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.175.csv')
# trip_frequency_vs_empty_colony_state(data, title='0.175')


# data = pd.read_csv('Data/Step_data/step_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.csv')






# data = pd.read_csv('Data/Interaction_data/interaction_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.csv')
#
# data = pd.read_csv('Data/space_inter_data.csv')
# # data = pd.read_csv('Data/order_inter_data.csv')
#
# df = data[data['colony state'] < 0.95]
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
# x = [0.1, 0.3, 0.5, 0.7, 0.9]
#
# df['bins'] = pd.cut(df['colony state'], bins=bins)
#
# int_no = df.groupby(['repeat','bins'])['colony state'].size().reset_index()
# int_no = int_no[int_no.repeat < 6]
# int_no.groupby('repeat')['colony state'].plot(use_index=False, style='-|', ms=10)
# plt.title('interaction number at different colony states')
# plt.ylabel('number of interactions')
# plt.xlabel('colony state')
# plt.ylim(0,2000)
# positions = (0,1,2,3,4)
# labels = (0.1, 0.3, 0.5, 0.7, 0.9)
# plt.xticks(positions, labels)
# plt.xlim(-.5, 4.5)
# plt.legend()
# plt.show()







# # crop at exit
# data = pd.read_csv('Data/Visit_data/visit_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.csv')

# data = pd.read_csv('Data/order_100_visit_data.csv')

# data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_visit_data.csv')
#
# df = data[data['trip length'] > 1]
# df = df[df['colony state'] < 0.95]
#
# trip_frequency_vs_empty_colony_state(df)
# plt.show()

# data = pd.read_csv('Data/space_vist_data.csv')
# #
# # data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_visit_data.csv')
# #
# # data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_visit_data.csv')
# #

# data = pd.read_csv('Data/space_vist_data.csv')
# data = pd.read_csv('Data/dettroph_average_velocity_visit_data.csv')
# df = data[data['trip length'] > 0]
# df = df[df['colony state'] < 0.95]
# bins = [x for x in np.arange(0, 1.1, 0.1)]
# df['bins'] = pd.cut(df['colony state'], bins=bins)
# #
# avg_exit = df.groupby('bins')['exiting crop'].mean()
# print(avg_exit)
# xx  = [i for i in np.arange(0.05, 1.05, 0.1)]
# #
# plt.plot(xx, avg_exit, '-o')
# #
# plt.ylim(0,1)
# plt.title('Average exiting crop of forager \n removed short trips')
# plt.ylabel('Foragers crop at exit')
# plt.xlabel('colony state')
# plt.show()


# x = df['colony state']
# y = df['exiting crop']
#
# plt.title('Exiting crop of forager')
# plt.ylabel('Foragers crop at exit')
# plt.xlabel('colony state')
# plt.scatter(x, y, s=1 )
# plt.show()







# # trip frequency and duration
# data = pd.read_csv('Data/Visit_data/visit_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.csv')

# data = pd.read_csv('Data/space_vist_data.csv')
# data = pd.read_csv('Data/order_vist_data.csv')

# data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_visit_data.csv')
#
# data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_visit_data.csv')
#

# data = pd.read_csv('Data/extrememove_visit_data.csv')
# data = data[data['trip length'] > 1]
# data = data[data['colony state'] < 0.95]

# bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
# df = data[data['colony state'] < 0.95]
# df['bin'] = pd.cut(df['colony state'], bins=bins)
# trip_dur = df.groupby('bin')['trip length'].mean()
# trip_frq = [1/x for x in trip_dur]
# x = [0.1, 0.3, 0.5, 0.7, 0.9]
# x_frq = x[::-1]
# plt.plot(x, trip_dur, '-o', c='blue', label='Simulation', zorder=5)


# x = np.linspace(0,0.95, 100)
# exp_dur = 1.4/(0.15*(1-x))

# plt.plot(x, exp_dur, c='red', linewidth=2, label='Theory')
# plt.ylim(0,90)
# plt.title('Trip duration simulation vs theory')
# plt.xlabel('Colony state')
# plt.ylabel('Trip duration')
# plt.legend()
# plt.show()





# plt.plot(trip_frq, x_frq, '-o')
# plt.show()



# # # Forager crop dynamics
# data = pd.read_csv('Data/Forager_data/forager_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.csv')
# data = pd.read_csv('Data/space_forag_data.csv')
# data = pd.read_csv('Data/oder_forag_data.csv')

# data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_forag_data.csv')
#
# data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_forag_data.csv')
#
# df = data[data.repeat < 3]
# # df.groupby('repeat')['crop'].plot(use_index=False, xlim=(0,13000))
# df.groupby('repeat')['crop'].plot(use_index=False)
# plt.title('Forager crop vs steps')
# plt.xlabel('step')
# plt.ylabel('Forager crop')
# plt.show()



# data = pd.read_csv('Data/space_vist_data.csv')
# data = pd.read_csv('Data/order_vist_data.csv')
# data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_visit_data.csv')
#
# data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_visit_data.csv')

# data = pd.read_csv('Data/extrememove_visit_data.csv')
# trip_frequency_vs_empty_colony_state(data, title='Simulation')

# x = np.linspace(0,0.95,100)
# exp_fre = (0.15*(1-x))/1.4
# plt.plot((1-x), exp_fre, c='red', linewidth=2, label ='Theory')
# plt.legend()
# plt.title('Trip frequency simulation vs theory')
# plt.show()


#
# # Perceived colony state vs colony state
# data_all = pd.read_csv('Data/Step_data/step_StochasticMovement_StochasticTrophallaxis_0.3_0'
#                     '.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.csv')
# data_int = pd.read_csv('Data/Interaction_data/interaction_StochasticMovement_StochasticTrophallaxis_0.3_0'
#                    '.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.csv')
#
# data_all = pd.read_csv('Data/space_step_data.csv')
# data_int = pd.read_csv('Data/space_inter_data.csv')

# data_all = pd.read_csv('Data/order_step_data.csv')
# data_int = pd.read_csv('Data/order_inter_data.csv')


# data_all = pd.read_csv('Data/partial_shuffle15_step_data.csv')
# data_int = pd.read_csv('Data/partial_shuffle15_inter_data.csv')

# data_all = pd.read_csv('Data/partial_shuffle10_step_data.csv')
# data_int = pd.read_csv('Data/partial_shuffle10_inter_data.csv')
#
#
#
# data_all = pd.read_csv('Data/space_200_step_data.csv')
# data_int = pd.read_csv('Data/space_200_inter_data.csv')
#
# data_all = pd.read_csv('Data/space_bias_shuffle_5_thresh0.25_thresh_0.75_step_data.csv')
# data_int = pd.read_csv('Data/space_bias_shuffle_5_thresh0.25_thresh_0.75_inter_data.csv')
#
#
# data_all = data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_step_data.csv')
# data_int = data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_inter_data.csv')

#
# bins = [x for x in np.arange(0,1.05,0.05)]
# df = data_all
# df['bin'] = pd.cut(df['colony state'], bins=bins)
# crop_state = df.groupby('bin')['crop'].mean()
#
#
# dff = data_int[data_int['ant type'] == 'Nestmate']
# ddf = data_int[data_int['ant type'] == 'Forager']
#
# dff = dff.set_index('step')
# ddf = ddf.set_index('step')
#
# dff['pre_int_crop'] = dff['forager/recipient crop'] - ddf['interaction volume']
#
# dff['bin'] = pd.cut(dff['colony state'], bins=bins)
# per_crop_state = dff.groupby('bin')['pre_int_crop'].mean()
#
# x = [0.1, 0.3, 0.5, 0.7, 0.9]
# x = [x for x in np.arange(0.05, 1.05, 0.05)]
#
# plt.plot(x, x, '-o', label='actual')
# plt.plot(x, per_crop_state, '-o', label='perceived')
# plt.title('Perceived vs actual colony state')
# plt.xlabel('Colony state')
# plt.ylabel('Perceived colony state')
# plt.xlim(0,1.1)
# plt.legend()
# plt.show()

# #




#  # Unloading rate


# data = pd.read_csv('Data/order_inter_data.csv')
# data = data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_inter_data.csv')
# #
# data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_inter_data.csv')
# data = data[data['ant type'] == 'Forager']
#
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
# df = data
# df['bin'] = pd.cut(data['colony state'], bins=bins)
#
# avg_vol_given = df.groupby('bin')['interaction volume'].mean()
#
# x = [0.1, 0.3, 0.5, 0.7, 0.9]
#
# plt.title('Unloading rate per colony state bin')
# plt.xlabel('Colony state')
# plt.ylabel(r'Unloading rate [$step^{-1}$]')
#
# plt.scatter(x, avg_vol_given)
#
# plt.show()


# data = pd.read_csv('Data/Visit_data/visit_StochasticMovement_StochasticTrophallaxis_0.3_0'
#                    '.15_homogenise_False_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.1.csv')
#
# trip_frequency_vs_empty_colony_state(data, title='0.175')
#
#
# data = pd.read_csv('Data/Visit_data/visit_StochasticMovement_StochasticTrophallaxis_0.3_0'
#                    '.15_homogenise_False_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.15.csv')
#
# trip_frequency_vs_empty_colony_state(data, title='0.15')
#
# data = pd.read_csv('Data/Visit_data/visit_StochasticMovement_StochasticTrophallaxis_0.3_0.15_homogenise_False_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.175.csv')
#
# trip_frequency_vs_empty_colony_state(data, title='0.1')
#
# plt.legend()
# plt.show()
# input()








# data = pd.read_csv('Data/Forager_data/forager_StochasticMovement_StochasticTrophallaxis_0.3_0'
#                    '.15_homogenise_False_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.25.csv')
#
#
# # # Position vs colony state
# df = data[data.index<20000]
# c = df['position']
# d = df['crop']
#
# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('position of forager', color=color)
# ax1.plot(c, color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('colony state', color=color)  # we already handled the x-label with ax1
# ax2.plot(d, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()



# # Trying to recreate exit probability plots

#
# data = pd.read_csv('Data/Forager_data/forager_StochasticMovement_StochasticTrophallaxis_0.3_0'
#                    '.15_homogenise_True_propagate_food_0_nestants_move_False_forager_lag_1_interaction_rate_1_inertia_0.5.csv')
# data['didexit'] = np.where(data['position'] == 0, 1, 0)
# data = data[data['colony state'] < 0.95]
# df = data.sample(int(0.05*len(data.index1)))

#
# data.to_csv('/home/gui/Agent-based-ants/Good_model/Data'
#             '/sampled_forager_data_for_probability_density_plot',
#             index=True,
#           index_label='index1')


# below_thresh = pd.DataFrame()
# for i in np.arange(0.01,1.01, 0.1):
#     below_thresh = below_thresh.append(data.loc[(data.crop < i).groupby([data.repeat, data.trip]).idxmax])
#
# below_thresh = below_thresh[below_thresh.repeat == 1]
#
# below_thresh['position'] = np.where(below_thresh.crop == 1, 0, 1)
# below_thresh['didexit'] = np.where(below_thresh.crop == 1, 1, 0)
#
# below_thresh['crop'] = [x if x != 1 else 0 for x in below_thresh.crop]
#
#
#
# below_thresh.to_csv('/home/gui/Agent-based-ants/Good_model/Data'
#             '/sampled_forager_data_for_probability_density_plot',
#             index=True,
#           index_label='index1')
# cropvscolonyexit(below_thresh)
# plt.show()
# Logistic fit for forager probability to exit as a function of her crop load and the colony state



# data = pd.read_csv('Data/oder_forag_data.csv')

# data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_forag_data.csv')
# data_int = data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_forag_data.csv')
# data = pd.read_csv('Data/inert_netmatesmoveandsharefood_lag100_sharerate1_bias_forag_data.csv')
#
# data.groupby('repeat')['colony state'].plot(use_index=False)
# plt.xlabel('step')
# plt.ylabel('Colony state')
# plt.title('Colony state progression over time')
# plt.show()



# derivative of the colony state

# data = pd.read_csv('Data/space_forag_data.csv')
# data = pd.read_csv('Data/order_step_data.csv')
#
#
# data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_step_data.csv')
#
# data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_step_data.csv')
#
# df = data[data['colony state'] < 0.95]
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
# x = [0.1, 0.3, 0.5, 0.7, 0.9]
#
# df['bin'] = pd.cut(df['colony state'], bins=bins)
# df['df/dx'] = data['colony state'].diff()
#
#
# x = df.groupby('bin')['colony state'].mean()
# y = df.groupby('bin')['df/dx'].mean()
#
# plt.title('Derivative of colony state vs colony state')
# plt.xlabel('Colony state')
# plt.ylabel('Derivative of colony state')
# plt.plot(x, y)
# plt.show()




# dataa = pd.read_csv('Data/order_vist_data.csv')
# data_int = pd.read_csv('Data/order_inter_data.csv')
#
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
# bins = [x for x in np.arange(0,1.05,0.05)]
# df = dataa
# df['bin'] = pd.cut(df['colony state'], bins=bins)
# dataa['bin'] = pd.cut(df['colony state'], bins=bins)
#
# dff = data_int[data_int['ant type'] == 'Nestmate']
# ddf = data_int[data_int['ant type'] == 'Forager']
#
# dff = dff.set_index('step')
# ddf = ddf.set_index('step')
#
# dff['pre_int_crop'] = dff['forager/recipient crop'] - ddf['interaction volume']
#
# dff['bin'] = pd.cut(dff['colony state'], bins=bins)
# this = pd.DataFrame(dff.groupby('bin')['pre_int_crop'].mean())
#
#
# def f(x):
#     return this.loc[x['bin']]
#
# hmm = dataa
# hmm['empty colony state'] = dataa.apply(f, axis=1)
# hmm['empty colony state'] = 1- hmm['empty colony state']
#
# trip_frequency_vs_empty_colony_state(hmm)
# plt.xlabel('Perceived empty colony state \n order model')
# plt.title('Trip frequency vs perceived colony state')
# plt.show()


# data = pd.read_csv('Data/netmatesmoveandsharefood_lag8_sharerate1_forag_data.csv')
#

# data = pd.read_csv('Data/space_leadingedgetest_forag_data.csv')

# data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_forag_data.csv')


# data_all = data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_forag_data.csv')
# #
# data = pd.read_csv('Data/netmatesmoveandsharefood_lag100_sharerate1_bias_forag_data.csv')
#
# data = data[data['le crop'] >=0]
# for i in range(max(data.repeat)):
#     df = data[data['repeat'] ==i]
#     x = df['colony state']
#     y = df['le crop']
#     plt.plot(x, y)
#
# plt.xlabel('Colony state')
# plt.ylabel('Forager crop at leading edge')
# plt.title('Forager crop at leading edge vs colony state \n leading edge where recipient crop below 0.9')
# plt.show()


# Number of ants before switch

# data = pd.read_csv('Data/extrememove_forag_data.csv')

# df = data[data.crop < 0.3]
# df = df[df['colony state']<0.95]

# ants_met = df.groupby(['repeat', 'trip']).size().tolist()
# ants_met_cs = df.groupby(['repeat', 'trip'])['colony state'].mean().tolist()

# x = np.linspace(0,0.95,100)
# exp_met = 0.7/(0.15*(1-x))

# plt.scatter(ants_met_cs, ants_met, c='blue', label='Simulation')
# plt.plot(x, exp_met, c='red', label='Theory', linewidth=5)
# plt.title('Number of ants met before bias switch vs simulation vs theory')
# plt.ylabel('Number of ants before switch')
# plt.xlabel('Colony state')
# plt.legend()
# plt.show()


# data = pd.read_csv('Data/dettroph_average_velocity_visit_data.csv')
# df = data[data['colony state'] < 0.95]
# #
# trip_frequency_vs_empty_colony_state(df, title='Average velocity model')
#
# y = [0.0572167, 0.051495, 0.0457733, 0.0400517, 0.03433, 0.0286083, 0.0228867, 0.017165, 0.0114433, 0.00572167]
# x = np.arange(1, 0, -0.1)
#
# plt.plot(x, y, '-o', label = 'Predicted')
# plt.legend()
# plt.show()


#
#
# data = pd.read_csv('Data/space_forag_data.csv')
# df = data[data['colony state'] < 0.95]
# df.name = 'space model'
#
#
# df = df.groupby('trip').mean()
# bins = np.linspace(0, 1, 20)
#
# df['bin'] = pd.cut(df['colony state'], bins=bins)
#
# binned_trips = df.groupby('bin').mean()
# x = np.linspace(0.05, 1, 19)
# y = binned_trips.position.tolist()
# plt.scatter(x,y)
# plt.title('Space model: Average position vs. Colony state \n binned by trips')
# plt.xlabel('Colony state')
# plt.ylabel('Average position')
# plt.show()
#
# def proportioanl(data, short_trips=True):
#     plt.style.use('ggplot')
#
#     bin_by  = input('What to bin by? (colony state or position): ') or 'colony state'
#
#     trips = data.groupby(['repeat', 'trip']).mean()
#     trips['len'] = data.groupby(['repeat', 'trip']).size()
#     if not short_trips:
#         trips = trips[trips.len > 6]
#
#
#     bins = np.linspace(min(trips[bin_by]), max(trips[bin_by]), 10)
#     trips['bin'] = pd.cut(trips[bin_by], bins)
#
#     binned_trips = trips.groupby('bin').mean()
#
#     dura = binned_trips.len.tolist()
#     freq = [1/j for j in dura]
#     cs = binned_trips['colony state'].tolist()
#     emp_cs = [1-x for x in cs]
#     one_over_empcs = [1/x for x in emp_cs]
#     pos = binned_trips.position.tolist()
#
#     # dic = {'dura': dura, 'freq': freq, 'cs': cs, 'emp_cs': emp_cs,
#     #        'one_over_empcs': one_over_empcs, 'pos': pos}
#     done = False
#     while not done:
#         fig,ax = plt.subplots()
#
#         answers = []
#         while len(answers) != 2:
#             questions = [
#                 inquirer.Checkbox('variables',
#                                   message="What would you like to plot?",
#                                   choices=['dura',  'freq', 'cs', 'emp_cs', 'one_over_empcs', 'pos'],
#                                   ),
#             ]
#             answers = inquirer.prompt(questions)['variables']
#             # print('Must select two choices!!')
#         xx = answers[0]
#         yy = answers[1]
#
#         x = locals()[xx]
#         y = locals()[yy]
#
#         print('Bin by ' + bin_by)
#         print('x: {}'.format(xx))
#         print('y: {}'.format(yy))
#
#         ax.scatter(x,y)
#         ax.set_xlabel(xx)
#         ax.set_ylabel(yy)
#         ax.set_title(xx + ' vs ' + yy)
#         fig.suptitle(df.name)
#         plt.draw()
#
#         questions = [
#             inquirer.Confirm('stop',
#                              message="Should I stop", default=True),
#         ]
#
#         done = inquirer.prompt(questions)['stop']
#         print(done)
#
#     plt.show()
#
# proportioanl(df, short_trips=True)
#
#



# pos_change_df = df[df['crop'].le(0.301)].groupby(['repeat', 'trip']).first()
#
# pos_change =  pos_change_df['position']
# pos_change_cs = pos_change_df['colony state']
#
# plt.scatter(pos_change_cs, pos_change)
#plt.show()

# bins = np.linspace(0,1,20)
#
# pos_change_df['bin'] = pd.cut(pos_change_df['colony state'], bins=bins)
#
# x = np.linspace(0.05, 0.95, 19)
# y = pos_change_df.groupby('bin')['position'].mean().tolist()
#
# #print(y)
#
# plt.scatter(x, y)
#plt.show()


# df = pd.read_csv('Data/space_forag_data.csv')
#
# df['above_threshold'] = np.where(df['crop'] > 0.3, 'yes', 'no')
# st_ant = df.groupby(['trip', 'above_threshold', 'position']).size().reset_index()
# st_ant = st_ant.rename({0:'times visited'}, axis=1)
# st_ant['times visited'] /= max(df.repeat)
#
# trying = st_ant.groupby(['above_threshold', 'position'])['times visited'].mean().reset_index()
#
# y = trying[trying['above_threshold'] == 'no']['times visited'].tolist()
# x = trying[trying['above_threshold'] == 'no']['position'].tolist()
#
# plt.scatter(x,y)
# plt.plot(range(len(x)), [1/0.25]*len(x))
# plt.show()
#
#
# y = trying[trying['above_threshold'] == 'yes']['times visited'].tolist()
# x = trying[trying['above_threshold'] == 'yes']['position'].tolist()
#
# y = y[:-2]
# x = x[:-2]
#
# plt.scatter(x,y)
# plt.plot(range(len(x)), [1/0.05]*len(x))
# plt.show()


# df = pd.read_csv('Data/dettroph_average_velocity_visit_data.csv')

# trip_frequency_vs_empty_colony_state(df, label='Simulation')

# x = np.arange(1, 0, -0.1)
# y = [0.0572167, 0.051495, 0.0457733, 0.0400517, 0.03433, 0.0286083, 0.0228867, 0.017165, 0.0114433, 0.00572167]

# plt.plot(x, y, '-o', label='Theory')
# plt.legend()
# plt.title('Trip frequency vs empty colony state')
# plt.xlabel('Empty colony state')
# plt.ylabel('Trip frequency')

# plt.show()
# print(os.path.dirname(os.path.realpath(__file__)))
# data = pd.read_csv('/home/gui/Agent-based-ants/Good_model/Data/dettroph_av_vel_visit_data.csv')
# # data = data[data['exiting crop'] > 0.3]
# # data = data[data['trip length'] > 1]
# df = data[data['colony state'] < 0.95]

# x, y = trip_frequency_vs_empty_colony_state(df)

# from scipy import stats
# # x = x[:-1]
# # y = y[:-1]
# slope, int, r, p, std = stats.linregress(x,y)

# y1 = x * slope
# y2 = x * (0.15/1.4)
# y3 = [i+ int for i in y2]
# plt.scatter(x, y, label='Simulation')
# plt.scatter(x, y1, label='Fitted')
# plt.scatter(x, y2, label='Theory')
# plt.plot(x, y3, 'o-', label='Theory with same intercept')
# plt.legend(title='Method')
# # plt.savefig('Data/slop_stoch.png')
# plt.show()
# print('Slop is {}, alpha/2k is {}'.format(slope, 0.15/1.4))
# print(r)



# data = pd.read_csv('Data/longrun_visit_data.csv')
#
# df= data
# df['short']= np.where(df['exiting crop'] < 0.3, 'no', 'yes')
# # df = df[df['short'] == 'yes']
#
# df['short_len'] = df.groupby('repeat')['step'].diff()
# df['short_freq'] = 1/df['short_len']
# df['freq'] = 1/df['trip length']
#
# bins = np.linspace(0,1,11)
#
# df['bin'] = pd.cut(df['empty colony state'], bins=bins)
#
# binned_df = df.groupby('bin').mean()
# binned_df['late_freq'] = 1/binned_df['short_len']
# x = binned_df['empty colony state'].tolist()
# y = binned_df['freq'].tolist()
# plt.scatter(x, y)
# # plt.ylim(0,1)
# plt.savefig('Data/short_trip_freq.png')



# TESTS ON DATA IN RUN CLASS

# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)
# plt.style.use('grayscale')
# import matplotlib.pyplot as plt


# def trip_freq(data):
#     model_data = data[data['colony state'] < 0.95]
#     model_data['empty colony state'] = model_data['empty colony state'].apply(
#         lambda j: round(j * 50) / 50)
#     model_data = model_data[model_data['empty colony state'] <= 0.9]
#     model_data = model_data.groupby('empty colony state').mean().reset_index()

#     def division_by_zero(a, b):
#         return 0 if a == 0 or b == 0 else a / b

#     y = model_data['trip length'].apply(
#         lambda x: division_by_zero(1, x)).to_numpy()

#     return y


# space_freq = trip_freq(data)
# norm_space_freq = np.array([x/space_freq.max() for x in space_freq])

# def percevied_vs_actual(data2, from_csv = False, title=None):
#     # bins = [x for x in np.arange(0,1.05,0.05)]
#     # df = data1
#     # df['bin'] = pd.cut(df['colony state'], bins=bins)
#     # crop_state = df.groupby('bin')['crop'].mean().to_numpy()
#     #
#     # dff = data2[data2['ant type'] == 'Nestmate']
#     # ddf = data2[data2['ant type'] == 'Forager']
#     #
#     # if from_csv:
#     #     dff = dff.set_index('step')
#     #     ddf = ddf.set_index('step')
#     # dff['pre_int_crop'] = dff['forager/recipient crop'] - ddf[
#     'interaction volume']
#     # dff['bin'] = pd.cut(dff['colony state'], bins=bins)
#     # per_crop_state = dff.groupby('bin')['pre_int_crop'].mean().to_numpy()
#     #
#     # return np.linalg.norm(crop_state-per_crop_state)
#
#     bins = [x for x in np.arange(0,1.05,0.05)]
#
#     dff = data2[data2['ant type'] == 'Nestmate']
#     ddf = data2[data2['ant type'] == 'Forager']
#
#     dff['pre_int_crop'] = dff['forager/recipient crop'] - ddf['interaction
#     volume']
#
#     dff['bin'] = pd.cut(dff['colony state'], bins=bins)
#     per_crop_state = dff.groupby('bin')['pre_int_crop'].mean()
#
#     x = [0.1, 0.3, 0.5, 0.7, 0.9]
#     x = [x for x in np.arange(0.05, 1.05, 0.05)]
#
#     plt.plot(x, x, '-o', label='actual')
#     plt.plot(x, per_crop_state, '-o', label='perceived')
#     plt.title('Perceived vs actual colony state \n' + title)
#     plt.xlabel('Colony state')
#     plt.ylabel('Perceived colony state')
#     plt.xlim(0,1.1)
#     plt.legend()
#     plt.show()
#
# # data1 = pd.read_csv('Data/space_step_data.csv')
# # data2 = pd.read_csv('Data/space_inter_data.csv')
#
# # space_pervsac = percevied_vs_actual(data1, data2, from_csv=True)
#
#


# def trip_durations(data):
#     bins = list(np.arange(0, 1, 0.05))
#     df = data
#     df['bin'] = pd.cut(df['colony state'], bins=bins)
#     trip_durations = df.groupby(['bin', 'trip length']).size().reset_index()

#     return trip_durations, bins


# def steps_on_ant(data):
#     df = data[data['crop'] > 0.3]
#     df = df[df['colony state'] < 0.95]
#     per_trip = df.groupby(['repeat', 'trip', 'position']).size().reset_index()
#     per_trip = per_trip.join(
#         df.groupby(['repeat', 'trip'])['colony state'].first(),
#         on=['repeat', 'trip'], rsuffix='_r')
#     per_trip = per_trip.rename({0: 'steps'}, axis=1)
#     bins = np.linspace(0, 1, 11)
#     per_trip['bin'] = pd.cut(per_trip['colony state'], bins=bins)
#     means = per_trip.groupby('bin').mean()
#     stdevs = per_trip.groupby('bin').std()
#     x = np.arange(0.05, 1, 0.1)

#     return x, means['steps'].tolist(), stdevs['steps'].tolist()


# def division_by_zero(a, b):
#     return 0 if a == b or b == 0 else a / b


# def func(x, a, b, c):
#     return a * x ** 2 + b * x + c


# def linear_test(data):
#     df = data[data['colony state'] < 0.95]
#     bins = np.linspace(0, max(df['empty colony state']), 11)
#     df['empcs_bin'] = pd.cut(df['empty colony state'], bins=bins)
#     df = df.groupby('empcs_bin').mean().reset_index()

#     x = df['empty colony state'].tolist()
#     y = df['trip length'].app;
#     y(lambda x: division_by_zero(1, x)).tolist()
#     popt, pcovs = curve_fit(func, x, y)
#     stds = np.sqrt(np.diag(pcovs))

#     return popt[0], stds[0]



# print("after function")
# ax.grid(False)
# ax.scatter(x,y)
# ax.errorbar(x, y, yerr=yerr, ls='none', capsize=6)
# ax.set_title('Steps on ants in stochastic model with nest \n size 1000 vs
# colony state')
# ax.set_xlabel('Colony state')
# ax.set_ylabel('Average steps on an ant')
# plt.show()


# fig, ax = plt.subplots()
# for i in np.arange(0.525, 1.05, 0.1):
#     ax.clear()
#     ax.grid(False)
#
#     x, y, yerr = f(bias_above = [1-i, 1], bias_below= [1,1.1])
#     avg = [1/(i - (1-i))] * len(x)
#     ax.scatter(x, y, label='Simulation')
#     ax.errorbar(x, y, yerr= yerr, ls='none')
#     ax.plot(x, avg, label='Predicted')
#
#     ax.set_title('Steps on ants by forager with average bias {} \n vs
#     colony state'.format(round(avg[0], 2)))
#     ax.set_xlabel('Colony state')
#     ax.set_ylabel('Steps on ants')
#     ax.legend()
#     fig.savefig('Steps_on_ant_with_bias_{}'.format(round(avg[0], 2)))

# fig, ax = plt.subplots()
#
# def ani(x, ax):
#     ax.clear()
#     ax.set_xlim(0, max(lengths['trip length']) + 5)
#     ax.set_ylim(0, max(lengths['count']) + 10)
#     sigh = lengths[lengths.index == inde[x]]
#     xx = [int(le) for le in sigh['trip length']]
#     y = [count for count in sigh['count']]
#     ax.bar(xx,y)
#     title = inde[x]
#     ax.set_title('Colony state interval {}'.format(title))
#

#
# for i in np.arange(0, 0.5, 0.2):
#     print('BACKWARDS PROBABILITY IS {}'.format(round(i,2)))
#     above = [round(i,2), 1]
#
#     lengths, cs_int = f(bias_above=above, bias_below=[1,1])
#     cs_int_r = [round(x,2) for x in cs_int]
#     lengths = lengths.set_index('bin')
#     lengths = lengths.rename({0:'count'}, axis=1)
#
#     inde = lengths.index.unique()
#
#     for i in range(0, len(inde), (len(inde)//3)):
#         fig1, ax1 = plt.subplots()
#         # fig1.suptitle('Backwards probability is {}'.format(above[0]),
#         fontsize=16)
#         ax1.set_xlim(0, max(lengths['trip length']) + 5)
#         ax1.set_ylim(0, max(lengths['count']) + 10)
#         sigh = lengths[lengths.index == inde[i]]
#         xx = [int(le) for le in sigh['trip length']]
#         y = [count for count in sigh['count']]
#         ax1.bar(xx, y)
#         title = inde[i]
#         ax1.set_title('Colony state interval {} \n Backwards probability is
#         {}'.format(title, above[0]))
#         plt.show()
#
#
#     fig, ax = plt.subplots()
#     fig.suptitle('Backwards probability is {}'.format(above[0]), fontsize=16)
#     ax.set_xlim(0, max(lengths['trip length'])+5)
#     ax.set_ylim(0, max(lengths['count'])+10)
#
#     animi = animation.FuncAnimation(fig, ani, fargs=(ax,), frames=len(
#     lengths.index.unique()),
#                                     interval=1000, blit=False, repeat=False)
#
#     animi.save('triplen_diffcs_backprob_{}.mp4'.format(above[0]),
#                writer='ffmpeg')

#    plt.show()


"""
    y = trip_freq(visit)
    # x = np.linspace(1, len(y) +1, len(y))
    x = np.linspace(0,1,len(y))
    yhat = savgol_filter(y, 25, 1)

    # plt.plot(yhat)
    # plt.plot(y)
    # plt.title('Trip frequency smoothed for bias: {}'.format(bias_above[0]))
    # plt.show()

    reg = linregress(x, yhat)
    grad = reg[0]
    std = reg[4]

    return grad, std

means = []
stdev = []
biases = []
string = []
for i in np.arange(0, 0.48, 0.03):
    print('ABOVE THRESH BIAS IS {}, BELOW THRESH BIAS IS {}'.format(round(i,
    2), 0.6))
    above = [round(i,2), 1]

    grad, std = f(bias_above=above, bias_below=[1, 1])

    string.append(str(above[0]))
    biases.append(above[0])
    stdev.append(std)
    means.append(grad)

# v = pd.read_csv('Data/space_vist_data.csv')
# y = trip_freq(v)
# yhat = savgol_filter(y, 25, 1)
# grad = np.gradient(yhat)
# men = np.mean(grad)
# va = statistics.stdev(grad)


means = means
stdev = stdev
x = np.linspace(1, len(means)+1, len(means))
plt.scatter(x, means, c='blue', label='Data')
# plt.errorbar(x, means, yerr=stdev, ls='none', ecolor='blue')

def f(bias, a):
    return (0.15/1.4) + (a * bias)

biases = np.array(biases)
popt, pcov = curve_fit(f, biases, means)
unknown = popt[0]

y = f(biases, unknown)

plt.scatter(x, y, label = r'Fitted data: $\frac{{0.15}}{{1.4}} \times cs + {} 
\times bias$'.format(round(unknown, 5)))
plt.title('Mean gradient for different biases')
plt.xticks(x, biases)
plt.legend()
plt.show()


"""

"""
DIFFERENCES BETWEEN TRIP FREQ AND DIFF BETWEEEN PERCEIVED AND ACTUAL CS

    if not visit.empty:
        # L2 norm of trip frequency
        run_freq = trip_freq(visit)
        norm_run_freq = np.array([x/run_freq.max() for x in run_freq])
        l2_norm = np.linalg.norm(norm_space_freq[:len(run_freq)] - 
        norm_run_freq)

        # Difference between perceived and actual cs
        diff = percevied_vs_actual(step, inter)
        a = 1
        return l2_norm, diff


norms = []
diffs = []
strings = []


        try:
            a, b = f(bias_above=above, bias_below=below)
            norms.append((a))
            diffs.append(b)
            # Can then look at gap between diff of space model and these models
            strings.append('A: ' + str(round(i, 1)) + ' B: ' + str(round(j, 
            1)))
        except:
            continue

x = np.linspace(1, len(norms)+1, len(norms))
space_stand = [space_pervsac for x in range(len(norms))]

ex_step = pd.read_csv('Data/extrememove_step_data.csv')
ex_inter = pd.read_csv('Data/extrememove_inter_data.csv')

ex_diff = percevied_vs_actual(ex_step, ex_inter, from_csv=True)

plt.scatter(10, ex_diff, c='orange')
plt.scatter(x, diffs, c= 'blue')
plt.plot(x, space_stand, color ='red')
plt.title('Difference between perceived and actual colony state \n in space 
model and models using various biases')
plt.xlabel('Biases')
plt.ylabel('Difference')
plt.xticks([x for x in range(len(strings))], [x[:6] for x in strings], 
rotation=70)
plt.tick_params(axis='x', which='major', labelsize=6)
plt.show()


ex_visit = pd.read_csv('Data/extrememove_visit_data.csv')

ex_freq = trip_freq(ex_visit)
norm_ex_freq = np.array([x/ex_freq.max() for x in ex_freq])
ex_l2_norm = np.linalg.norm(norm_space_freq[:len(norm_ex_freq)] - norm_ex_freq)

plt.scatter(10, ex_l2_norm, c = 'red')
plt.scatter(x, norms, c= 'blue')
plt.title('Difference between normalised trip frequencies for different 
biases \n compared to space model')
plt.xlabel('Biases')
plt.ylabel('Difference')
plt.xticks([x for x in range(len(strings))], [x[:6] for x in strings], 
rotation=70)
plt.tick_params(axis='x', which='major', labelsize=6)
plt.show()


"""

"""
COMPARING EXITING CROP STATE AT THE DIFFERENT COLONY STATES
    df = a.all_visit_data
    if not df.empty:
        df = df[df['trip length'] > 1]
        df = df[df['colony state'] < 0.95]
        bins = [x for x in np.arange(0, 1.1, 0.1)]
        df['bins'] = pd.cut(df['colony state'], bins=bins)

        avg_exit = df.groupby('bins')['exiting crop'].mean().tolist()

    else:
        return None, None

    return  avg_exit, np.mean(avg_exit)


avg_exit_list = []
av_exit = []
strings = []
avg_exit_dic = {}

fig, ax = plt.subplots()

for i in np.arange(0.05, 1, 0.1):
    for j in np.arange(0.05, 1, 0.1):

        print('ABOVE THRESH BIAS IS {}, BELOW THRESH BIAS IS {}'.format(i, j))
        above = [round(i,1), round(1-i,1)]
        below = [round(j,1), round(1-j,1)]
        a, b = f(bias_above=above, bias_below=below)

        avg_exit_list.append(a)
        av_exit.append(b)
        strings.append('A: ' + str(round(i,1)) + '\n' + ' B: ' + str(round(j,
        1)))
        avg_exit_dic['A: {} B: {}'.format(round(i,1), round(j,1))] = a

        print(a, b)

        if b is not None:
            if b > 0:
                normal = pl.Normalize(0, 1)
                color = pl.cm.plasma(normal(b))

                s = ax.scatter(i, j, c=color)


av_print_list = [x for x in avg_exit_list if x is not None and np.mean(x) > 0]
print('\n'.join(' '.join(map(str,sl)) for sl in av_print_list))
avg_exit_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,
v in avg_exit_dic.items() ]))
avg_exit_df.to_csv('Average_exit_values.csv')

s.set_clim([0, 1])
cb = fig.colorbar(s)
ax.set_xlabel('Above average bias')
ax.set_ylabel('Below average bias')
ax.set_title('Phase space of average exiting crop with \n different biases')
ax.set_xlim(0,1.1)
plt.show()



y = [x for x in av_exit if x is not None and  x > 0]
sd = [statistics.stdev(x) for x in avg_exit_list if x is not None and 
np.mean(x) > 0 ]

x = np.linspace(1, len(y) +1, len(y))

plt.scatter(x, y)
plt.errorbar(x, y, yerr=sd, ls='none')

plt.title('Average exiting crop at each bias combo \n and stdev')
plt.xlabel('Bias combo')
plt.ylabel('Average exiting crop')
plt.xticks(x, strings, rotation=45)
plt.show()

"""
