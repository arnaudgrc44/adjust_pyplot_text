import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import copy


# PURPOSE :
# goal of this module is to arrangle text annotation in pyplot 


class Rectangle:
	""" class of the geometrical figure of the plate rectangle, i.e
	the sides of the rectange are collinear with the canonical basis vectors
	(1,0) and (0,1)"""
	def __init__(self, x1, l, h):
		"""x1 for the coordinates of the bottom left point.
		l for the length of the rectangle to get the bottom right point
		and h for the height of the rectangle. This is a "plate" rectangle"""
		if not isinstance(x1, np.ndarray):
			raise TypeError("x1 must be numpy.ndarray")
		if x1.shape != (2,):
			raise TypeError("x1 must be shape (2,)")
		if not isinstance(l, (float, int)):
			raise TypeError("l must be float or int")
		if not isinstance(h, (float,  int)):
			raise TypeError("h must be float or int")
		self.x1 = x1
		self.l = l
		self.h = h
		self.x2 = x1 + np.array([l, 0])
		self.y1 = x1 + np.array([0, h])
		self.y2 = x1 + np.array([l, h])
	
	def isEqual(self, r):
		"""return True if self rectangle equals passed argument rectangle"""
		if not isinstance(r, Rectangle):
			raise TypeError("r must be class Rectangle")
		if (self.x1 == r.x1).all() and self.l == r.l and self.h == r.h:
			return(True)
		else:
			return(False)
	
	def contains_point(self, p):
		"""method that evaluates if self rectangle containsa point
		or not. 
		input : numpy.ndarray of shape (2,)
		output : bool
		"""
		if not isinstance(p, np.ndarray):
			raise TypeError("p must be numpy.ndarray")
		if p.shape != (2,):
			raise TypeError("p must be shape (2,)")
		if (p[0] >= self.x1[0]) and (p[0] <= self.x2[0]) and\
			(p[1] >= self.x1[1]) and (p[1] <= self.y1[1]):
			return(True)
		else:
			return(False)
		
	def contains_rectangle_top(self, r):
		"""method that evaluates if self rectangle contains at least
		one of another's rectangle tops or not. 
		input : object of class Rectangle
		output : bool
		"""
		if not isinstance(r, Rectangle):
			raise TypeError("r must be class Rectangle")
		if self.contains_point(r.x1) or self.contains_point(r.x2)\
			or self.contains_point(r.y1) or self.contains_point(r.y2):
			return(True)
		else:
			return(False)
	
	def covers_rectangle(self, r):
		"""method that evaluates if self rectangle covers another
		rectangle or not. Return true if self and rectangle doesn't contains
		each other one of theirs top.
		input : object of class Rectangle
		output : bool
		"""
		if not isinstance(r, Rectangle):
			raise TypeError("r must be class Rectangle")
		if self.contains_rectangle_top(r) or r.contains_rectangle_top(self):
			return(True)
		else:
			return(False)

class TextRectangle:
	"""class of particular rectangle in the context of pyplot.
	For the need of our problematic which is to avoid texts annotation
	overlayering.
	
	a TextRectngle is just a rectangle with a state. A state is a given 
	position compared to the point at wich the rectangle (i.e the text 
	in pyplot) is attached.
	
	We consider 4 possible state. The first state is the default state
	which is the text positionned at the top-right of the attached point.
	Second state the text is at the top left, in the third state it's at the
	right bottom and at the fourth state it's at the left bottom.
	
	Given this, estimate the length is important to avoid an important shift
	for the second and fourth states."""
	
	def __init__(self, r, state, marge=np.array([0,0])):
		if not isinstance(r, Rectangle):
			raise TypeError("r must be a Rectangle")
		if not isinstance(marge, np.ndarray):
			raise TypeError("marge must be numpy array")
		if not marge.shape == (2,):
			raise TypeError("marge must be shape (2,)")
		if state not in [1, 2, 3, 4]:
			raise TypeError("state must be 1, 2, 3 or 4")
		self.r = r
		self.state = state
		self.marge = marge
	
	def left_translation(self):
		x1 = self.r.x1 + np.array([-self.r.l - 2*self.marge[0], 0] )
		self.r = Rectangle(x1, self.r.l, self.r.h)
		self.state += 1
	
	def right_translation(self):
		x1 = self.r.x1 + np.array([self.r.l + 2*self.marge[0], 0])
		self.r = Rectangle(x1, self.r.l, self.r.h)
		self.state -= 1
	
	def upper_translation(self):
		x1 = self.r.x1 + np.array([0, self.r.h + 2*self.marge[1]])
		self.r = Rectangle(x1, self.r.l, self.r.h)
		self.state -= 2
	
	def down_translation(self):
		x1 = self.r.x1 + np.array([0, -self.r.h - 2*self.marge[1]])
		self.r = Rectangle(x1, self.r.l, self.r.h)
		self.state += 2
	
	def change_state(self, state):
		if state not in [1, 2, 3, 4]:
			raise TypeError("state must be 1, 2, 3 or 4")
		if state == self.state:
			return(0)
		if self.state == 1:
			if state == 2:
				self.left_translation()
			if state == 3:
				self.down_translation()
			if state == 4:
				self.down_translation()
				self.left_translation()	
		if self.state == 2:
			if state == 1:
				self.right_translation()
			if state == 3:
				self.down_translation()
				self.right_translation()
			if state == 4:
				self.down_translation()
		if self.state == 3:
			if state == 1:
				self.upper_translation()
			if state == 2:
				self.left_translation()
				self.upper_translation()
			if state == 4:
				self.left_translation()
		if self.state == 4:
			if state == 1:
				self.upper_translation()
				self.right_translation()
			if state == 2:
				self.upper_translation()
			if state == 3:
				self.right_translation()
		return(1)	


class CloudOfTextRectangle:
	"""class to represent an esemble of TextRectangle object. I.e in the
	pyplot context, a cloud of text annotation to attached point. Goal is to 
	arrange the cloud of point in order to have the least conflicts between 
	ext (i.e Rectangles). 
	
	So we will define a method to get and count the conflicts, an other to
	arrange the cloud of text rectangle in all possible ways with the "error"
	function (number of conflicts) to minimize"""
	
	def __init__(self, list_of_tr):
		# check input type
		if not isinstance(list_of_tr, list):
			raise TypeError("argument must be list")
		for r in list_of_tr:
			if not isinstance(r, TextRectangle):
				raise TypeError("list element must be all TextRectangle, received {} instead".format(str(type(r))))
		self.list_tr = list_of_tr
		self.conflicts = None
		self.get_conflicts()
	
	def get_conflicts(self):
		"""function that compute the conflicts associated to a cloud of
		text rectangles.
		output : list of tuple. tuple or text rectangle's index involved
		in conflicts. So conflicts are modelised as pairs."""
		conflicts = []
		for i in range(len(self.list_tr)-1):
			for j in range(i+1, len(self.list_tr)):
				if self.list_tr[i].r.covers_rectangle(self.list_tr[j].r):
					conflicts.append((i, j))
		self.conflicts = conflicts
		return(0)
	
	
	def new_config_cloud(self, index, state):
		""" function to change the state of i-ème text rectangle in the
		cloud of text rectangle."""
		new_config = copy.deepcopy(self.list_tr)
		new_config[index].change_state(state)
		return(CloudOfTextRectangle(new_config))
	
		
	def treat_conflicts(self, parent_nodes_conflicts, cpt):
		"""
		input : empty list. Needed for recursion.
		output : a tree of resolved conflicts in the form of
			 recursively nested dictionary.
		
		resolution of the conflicts is recursive. First, we try
		to resolve the first conflict of the cloud of text rectangle.
		
		first resolution's try gives two list : a list of cloud with
		new configuration (the textRectangle has a different state's
		configuration) and without the first conflict (i.e the confluct 
		resolved), and a list of cloud with new new configuration and new 
		conflict. The we recursely do this resolution on the two different
		kind of clouds we get. The stop criteria is no conflict, or if we 
		have previously explore all the conflict/configuration situation
		before."""
		# check input type
		if not isinstance(parent_nodes_conflicts, list):
			raise TypeError("parent_nodes must be list")
		
		n_conflict = len(self.conflicts)
		# stop condition to recursion --> no conflict in the cloud
		if n_conflict == 0 or cpt > 3 :
			return({"parent": self, "childrens": None})
		parent_nodes_conflicts.append(self.conflicts)
		n_min = min([len(c) for c in parent_nodes_conflicts])
		#print("parent_nodes_conflicts:")
		#print(parent_nodes_conflicts)
		configs = []
		first_conflict = self.conflicts[0]
		for s in [1, 2, 3, 4]:
			configs.append(self.new_config_cloud(first_conflict[0], s))
			configs.append(self.new_config_cloud(first_conflict[1], s))
		new_configs_better = [c for c in configs if 
				(c.conflicts not in parent_nodes_conflicts) 
						and (len(c.conflicts) < n_min)]
		new_configs_even = [c for c in configs if 
				(c.conflicts not in parent_nodes_conflicts) 
						and (len(c.conflicts) == n_min)]
		#new_configs_better = [c for c in configs if 
		#		(c.conflicts not in parent_nodes_conflicts) 
		#				and (len(c.conflicts) < n_conflict)]
		#new_configs_even = [c for c in configs if 
		#		(c.conflicts not in parent_nodes_conflicts) 
		#				and (len(c.conflicts) == n_conflict)]
		# size limitation to four childrens
		new_configs = new_configs_better + new_configs_even[:max(0, 4-len(new_configs_better))]
		#print("new_config")
		#print([c.conflicts for c in new_configs])
		# second stop condition : no more config not explored and
		# no config found with new conflict to treat
		if len(new_configs) == 0:
			return({"parent": self, "childrens": None})
		# compteur qui compte les tentatives "infructeurses completes de trouver
		# un sous chemin meilleurs. L'objectif est d'éviter les boucles trop goourmandes 
		# en calcul
		if sum([len(c.conflicts) == n_conflict for c in new_configs]) == len(new_configs):
			cpt += 1
		else:
			cpt = 0
		childrens = [c.treat_conflicts(parent_nodes_conflicts, cpt) for c in new_configs]
		return({"parent": self, "childrens": childrens})
	
	# main function to arrange texts using treat_conflicts result	
	def arrange_text(self, arrows=False):
		if arrows:
			for i, tr in enumerate(self.list_tr):
				if tr.r.x1[0] >= 0 and tr.r.x1[1] >= 0:
					pass
				if tr.r.x1[0] < 0 and tr.r.x1[1] >= 0:
					self.list_tr[i].change_state(2)
				if tr.r.x1[0] >= 0 and tr.r.x1[1] < 0:
					self.list_tr[i].change_state(3)
				if tr.r.x1[0] < 0 and tr.r.x1[1] < 0:
					self.list_tr[i].change_state(4)
		resolve_conflicts_tree = self.treat_conflicts([], cpt=0)
		def get_tree_leaves(tree):
			if not isinstance(tree, dict):
				raise TypeError("input must be dict")
			if tree["childrens"] is None:
				return([tree["parent"]])
			else:
				return([get_tree_leaves(c) for c in tree["childrens"]])
		tree_leaves = get_tree_leaves(resolve_conflicts_tree)
		leaves = []
		def flatten(list_of_list):
			for el in list_of_list:
				if not isinstance(el, list):
					leaves.append(el)
				else:
					flatten(el)
		flatten(tree_leaves)
		sorted_leaves = sorted(leaves, key=lambda x:\
						 len(x.conflicts), reverse=False)
		self.list_tr = sorted_leaves[0].list_tr
		self.get_conflicts()
		return(0)


#my_tree_result = make_cloud_tree(resolve_conflicts_tree2)	
#tree_leaves = my_tree_result.leaves
#sorted_leaves = sorted(tree_leaves, key=lambda x: len(x.value.conflicts), reverse=False)


def make_text_rectangle(xy, text, marge, ax, coef=1):
	"""function to create text rectangle from xy coordonnee and text to print.
	Added a coef to balance output for some reason"""
	if not isinstance(text, str):
		raise TypeError("text input must be str")
	x_scope = ax.get_xlim()[1] - ax.get_xlim()[0]
	y_scope = ax.get_ylim()[1] - ax.get_ylim()[0]
	figwidth = ax.get_figure().get_figwidth()
	figheight = ax.get_figure().get_figheight()
	sizing_dict = {"default": 1.1, "a": 1.1, "b": 1.1, "c": 1, "d": 1.1, "e": 1.1, "f": 0.65, "g": 1.1, "h": 1.1, "i": 0.55, "j": 0.55, "k": 1.1, "l": 0.55, "m": 1.54, "n": 1.1, "o": 1.1, "p": 1.1, "q": 1.1, "r": 0.825, "s": 1, "t": 0.65, "u": 1.1, "v": 1.1, "w": 1.3, "x":1.1, "y": 1.1, "z": 1, "_": 0.9, "'": 0.55, "1": 1.15, "2": 1.15, "3": 1.15, "4": 1.15, "5": 1.15, "6": 1.15, "7": 1.15, "8": 1.15, "9": 1.15, "0": 1.15}
	# as we compute character size in a x_scope of length 16 and figwidth of 10
	#, we have to rescale the character's size to the targeted figure scale.
	for key in sizing_dict.keys():
		sizing_dict[key] = sizing_dict[key] * x_scope / (figwidth * 10)
	h = 1.5 * x_scope / (figwidth * 10)
	l = 0
	for c in text.lower():
		if c not in list(sizing_dict.keys()):
			#raise ValueError("character {} in text not recognize by sizing\
#dict. Only lower character without accent are accepted, and only _ and ' are\
#accepted for punctuation".format(c))
			l += sizing_dict["default"]
		else:
			l += sizing_dict[c]
	#print("text : {0}, size : {1}".format(text, str(l)))
	return(TextRectangle(Rectangle(xy + marge, l, h), 1, marge))


def adjust_text(ax, add_marge=True, arrows=False):
	if len(ax.texts) == 0:
		return(0)	
	# if the axis aspect is set to equal to keep proportion (for cercle) for example, the x_scope is going to be multiply by two when we will enlarge 
	#if ax.get_aspect() == "equal":
	#	x_scope *= 2
	x_scope = ax.get_xlim()[1] - ax.get_xlim()[0]
	y_scope = ax.get_ylim()[1] - ax.get_ylim()[0]
	figwidth = ax.get_figure().get_figwidth()
	figheight = ax.get_figure().get_figheight()
	if add_marge:
		marge = np.array([x_scope/figwidth, y_scope/figheight]) / 15
	else:
		marge = np.array([0, 0])
	list_tr = []
	for text in ax.texts:
		list_tr.append(make_text_rectangle(np.array(text.xyann),\
							text.get_text(), marge, ax))
	cloud = CloudOfTextRectangle(list_tr)
	cloud.arrange_text(arrows)
	for i in range(len(ax.texts)):
		tr = cloud.list_tr[i]
		ax.texts[i].set_x(tr.r.x1[0])
		ax.texts[i].set_y(tr.r.x1[1])





######## TEST ########
def test_rectangle_class(marge=0.1):
	x1 = np.array([1, 2])
	r1 = Rectangle(x1, 3, 1)
	r2 = Rectangle(x1 + np.array([1, 0]), 3, 1)
	r3 = Rectangle(x1 + np.array([1, 3]), 3, 1)
	if not r1.contains_point(x1):
		raise AssertionError("contains_point doesn't work propelly")
	if r1.contains_point(x1 + np.array([-1, 0])):
		raise AssertionError("contains_point doesn't work propelly")
	if not r1.contains_rectangle_top(r2):
		raise AssertionError("contains_rectangle_top doesn't work propelly")
	if r1.contains_rectangle_top(r3):
		raise AssertionError("contains_rectangle_top doesn't work propelly")
	if not r1.covers_rectangle(r2):
		raise AssertionError("covers_rectangle doesn't work propelly")
	if r1.covers_rectangle(r3):
		raise AssertionError("covers_rectangle doesn't work propelly")
	r1.juxtapose_rectangle(r2)
	expected_rect = Rectangle(x1 - np.array([2.1, 0]), 3, 1)
	if not r1.isEqual(expected_rect):
		raise AssertionError("juxtapose_rectangle doesn't work propelly")
	r1.juxtapose_rectangle(r3)
	if not r1.isEqual(expected_rect):
		raise AssertionError("juxtapose_rectangle doesn't work propelly")
	return(0)

# test Rectangle class
#test_rectangle_class(marge=0.1)


def test_class_cloud():
	# def rectangles
	x1 = np.array([2, 2])
	r1 = Rectangle(x1, 2, 1)
	r2 = Rectangle(x1 + np.array([1, 0]), 2, 1)
	r3 = Rectangle(x1 + np.array([-3, 0]), 2, 1)
	r4 = Rectangle(x1 + np.array([0, -1]), 2, 0.5)
	x5 = np.array([7, 4])
	r5 = Rectangle(x5, 3, 2)
	r6 = Rectangle(x5 + np.array([2, 1]), 2, 1)
	# def text ractangles
	tr1 = TextRectangle(r1, 1)
	tr2 = TextRectangle(r2, 1)
	tr3 = TextRectangle(r3, 1)
	tr4 = TextRectangle(r4, 1)
	tr5 = TextRectangle(r5, 1)
	tr6 = TextRectangle(r6, 1)
	# def clouds
	cloud1 = CloudOfTextRectangle([tr1, tr2, tr3])
	cloud2 = CloudOfTextRectangle([tr1, tr2, tr3, tr4, tr5, tr6])
	# resolve conflicts
	resolve_conflicts_tree1 = cloud1.treat_conflicts([], 0)
	resolve_conflicts_tree2 = cloud2.treat_conflicts([], 0)
	# arrange_text
	cloud2.conflicts
	cloud2.arrange_text()
	cloud2.conflicts
	return(0)

#test_class_cloud()

def example(adjust = False):
	np.random.seed(0)
	texts = ["Coca_cola", "IBM", "Microsoft", "Google", "General_Electric",
			"Mc_Donalds", "Intel", "Nokia", "Disney", "HP", "Toyota",
			"Mercedes-benz", "Gilette", "Cisco", "BMW", "Apple", "Marlboro",
			"Samsung", "Honda", "H&M", "Oracle", "Pepsi", "Nike", "SAP",
			"Nescafe", "IKEA", "Jp_Morgan", "Budweiser", "UPS", "HSBC", "Canon",
			"Sony", "Kellog's", "Amazon", "Goldman_Sachs", "Nintendo", "DELL",
			"Ebay", "Gucci", "LVMH", "Heinz", "Zara", "Siemens", "Netflix",
			"Louis_Vuitton", "Channel", "Facebook", "Tesla", "Spotify",
			"Porsche", "Starbucks", "UBER", "KFC", "Linkedin"]
	x = np.random.uniform(-10, 10, len(texts))
	y = np.random.uniform(-10, 10, len(texts))
	fig, ax = plt.subplots()
	ax.scatter(x, y, marker="+", s=30)
	plt.xlim(-(x.max()+3), x.max()+3)
	plt.ylim(-(y.max()+3), y.max()+3)
	for i, txt in enumerate(texts):
		ann = txt
		ax.annotate(ann, xy=(x[i], y[i]), xytext=(x[i], y[i]))
	if adjust:
		adjust_text(ax)
	plt.show()

#example(False)
#example(True)
