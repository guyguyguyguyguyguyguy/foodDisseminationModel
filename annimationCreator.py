import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colorbar as cbar
import pylab as pl
import tkinter as tk
import re
import matplotlib.animation as animation
import ast
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
# writer=animation.FFMpegFileWriter(fps=3, bitrate=150)
writer = animation.writers['ffmpeg'](fps=max(2, 3), bitrate=2000)	

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["plum", "royalblue"])


class Animate:

	def __init__(self, save_name=False, ant=False):
		self._df_file = self.get_data()
		self._df_file = '~/Agent-based-ants/Good_model/Data/singlerun_step_data.csv' if self._df_file is None else self._df_file;
		self.save_name = save_name
		self.ant = ant
		self.validate()
		self.fig, self.ax = plt.subplots(figsize=(25,5))
		normal = pl.Normalize(0, 1)
		self.cax, _ = cbar.make_axes(self.ax)
		self.cb2 = cbar.ColorbarBase(self.cax, cmap=cmap, norm=normal)

		if re.search("[$.](.+)", self._df_file).group(1) == 'csv':
			self.df = pd.read_csv(self._df_file)
		elif re.search("[$.](.+)", self._df_file).group(1) in ['xls', 'xlsx', 'xlsm', 'xlsb']:
			self.df = pd.read_excel(self._df_file)

		self.repeat_df = self.df[self.df.id <=23]
		# self.repeat = int(input("Which repeat to animate? "))
		# self.repeat_df = self.df[self.df.repeat == self.repeat]

	@staticmethod
	def get_data():
		root = tk.Tk()
		root.eval('tk::PlaceWindow %s center' % root.winfo_pathname(root.winfo_id()))
		root.withdraw()
		file = tk.filedialog.askopenfile(
		    parent=root, mode='rb', title='Choose a file')
		if file != None:
			data = file.read()
			file.close()
			return str(file.name)
		return None 

	def animation(self):
		if self.save_name:
			ani = animation.FuncAnimation(
			    # self.fig, self.__animate_repeat, interval=1, frames=max(self.repeat_df.step))
			    self.fig, self.__animate_repeat, interval=1, frames=(500))

			ani.save(self.save_name, writer=writer)

		else:
			for i in range(10):
				self.fig, self.ax = plt.subplots(figsize=(41,5))
				normal = pl.Normalize(0, 1)
				self.cax, _ = cbar.make_axes(self.ax)
				self.cb2 = cbar.ColorbarBase(self.cax, cmap=cmap, norm=normal)
				self.__animate_repeat(i)
				plt.show(block=False)
				plt.pause(0.001)
				plt.close()

	def __animate_repeat(self, x):
		self.ax.clear()
		self.ax.add_patch(pl.Rectangle([0, -0], 1, 3, color='forestgreen'))
		self.ax.grid(which='major', axis='both',
		             linestyle='-', color='k', linewidth=1)
		self.ax.set_xticks(np.arange(0, self.repeat_df['position'].max(), 1))
		self.ax.set_yticks(np.arange(0, 2, 1))

		step_df = self.repeat_df[self.repeat_df.step == x+1]
		nest = step_df[step_df.id != 0]
		forag = step_df[step_df.id == 0]

		nes_pos = nest.position.tolist()
		nes_crop = nest.crop.tolist()

		for_pos = forag.position.tolist()
		for_crop = forag.crop.tolist()
		self.__animate_nestmates(nes_pos, nes_crop)
		self.__animate_foragers(for_pos, for_crop)


	def __animate_nestmates(self, pos, crop):


		normal = pl.Normalize(0, 1)
		colors = cmap(normal(crop))
    
		for p,c in zip(pos,colors):
			if not isinstance(pos, str):
				rect = pl.Rectangle([p,0],1,3,color=c)
				self.ax.add_patch(rect)
			else:
				pos = [ast.literal_eval(x) for x in pos]
				rect = pl.Rectangle(pos,1,1,color=c)
				self.ax.add_patch(rect)


	def __animate_foragers(self, pos, crop):

			for p in pos:
				if not isinstance(pos, str):
					if self.ant:
						try:
							image = 'ant.png'
							ab = AnnotationBbox(self.__getImage(image), (p+0.5, 0.5),  frameon=False)
							self.ax.add_artist(ab)
						except:
							# rect = pl.Rectangle([p,0],1,1,color='red')
							# self.ax.add_patch(rect)
							self.ax.scatter(p+0.5, 1.5, c='pink', s=500,zorder=10)


					else:
						# rect = pl.Rectangle([p,0],1,1,color='red')
						# self.ax.add_patch(rect)
						self.ax.scatter(p+0.5, 1.5, c='pink', s=500,zorder=10)
				else:
					# pos = [ast.literal_eval(x) for x in pos]
					pos = ast.literal_eval(pos)		
					if self.ant:
						try:
							image = 'ant.png'
							ab = AnnotationBbox(self.__getImage(image), (p-1 + 1.5, 0 +0.5),  frameon=False)
							self.ax.add_artist(ab)
						except:
							rect = pl.Rectangle(pos,1,1,color='red')
							self.ax.add_patch(rect)

					else:
						rect = pl.Rectangle(pos,1,1,color='red')
						self.ax.add_patch(rect)


	@staticmethod
	def __getImage(path):
		return OffsetImage(plt.imread(path), zoom=0.075)


	def validate(self):
		assert 'step' in self._df_file, "Must be a step dataframe."


an = Animate(save_name='video_of_ants.mp4', ant=True)
an.animation()
