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
	
		
	def treat_conflicts(self, parent_nodes_conflicts):
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
		if n_conflict == 0:
			return({"parent": self, "childrens": None})
		parent_nodes_conflicts.append(self.conflicts)
		configs = []
		first_conflict = self.conflicts[0]
		for s in [1, 2, 3, 4]:
			configs.append(self.new_config_cloud(first_conflict[0], s))
			configs.append(self.new_config_cloud(first_conflict[1], s))
		new_configs = [c for c in configs if (c.conflicts not in 
					parent_nodes_conflicts) and (len(c.conflicts)<= n_conflict)]
		# second stop condition : no more config not explored and
		# no config found with new conflict to treat
		if len(new_configs) == 0:
			return({"parent": self, "childrens": None})
		childrens = [c.treat_conflicts(parent_nodes_conflicts) for c in 
															new_configs]
		return({"parent": self, "childrens": childrens})
	
	# main function to arrange texts using treat_conflicts result	
	def arrange_text(self):
		resolve_conflicts_tree = self.treat_conflicts([])
		def get_tree_leaves(tree):
			if not isinstance(tree, dict):
				raise TypeError("input must be dict")
			if tree["childrens"] is None:
				return(tree["parent"])
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


def make_text_rectangle(xy, text, marge, x_scope, y_scope, coef=1):
	"""function to create text rectangle from xy coordonnee and text to print.
	Added a coef to balance output for some reason"""
	if not isinstance(text, str):
		raise TypeError("text input must be str")
	# as we compute character size in a x_scope of length 16, we have to
	# rescale the character's size to the targeted figure scale.
	scale_x = x_scope/16
	h = y_scope/16 * 0.3
	sizing_dict = {"a": 0.1, "b": 0.1, "c": 0.09, "d": 0.1, "e": 0.1, "f": 0.06,
				"g": 0.1, "h": 0.1, "i": 0.05, "j": 0.05, "h": 0.1, "l": 0.05,
				"m": 0.14, "n": 0.1, "o": 0.1, "p": 0.1, "q": 0.1, "r": 0.075,
				"s": 0.09, "t": 0.06, "u": 0.1, "v": 0.1, "w": 0.12, "x":0.1,
				"y": 0.1, "z": 0.09, "_": 0.08, "'": 0.05, "1": 0.1, "2": 0.1,
				"3": 0.1, "4": 0.1, "5": 0.1, "6": 0.1, "7": 0.1, "8": 0.1,
				"9": 0.1, "0": 0.1}
	for key in sizing_dict.keys():
		sizing_dict[key]*=(scale_x*coef)
	l = 0
	for c in text.lower():
		if c not in list(sizing_dict.keys()):
			#raise ValueError("character {} in text not recognize by sizing\
#dict. Only lower character without accent are accepted, and only _ and ' are\
#accepted for punctuation".format(c))
			l += 0.1
		else:
			l += sizing_dict[c]
	#print("text : {0}, size : {1}".format(text, str(l)))
	return(TextRectangle(Rectangle(xy + marge, l, h), 1, marge))


def adjust_text(ax, add_marge=True):
	if len(ax.texts) == 0:
		return(0)	
	x_scope = ax.viewLim.xmax - ax.viewLim.xmin
	y_scope = ax.viewLim.ymax - ax.viewLim.ymin
	if add_marge:
		marge = np.array([x_scope, y_scope])/400
	else:
		marge = np.array([0, 0])
	list_tr = []
	for text in ax.texts:
		list_tr.append(make_text_rectangle(np.array(text.xyann),\
							text.get_text(), marge,  x_scope, y_scope))
	cloud = CloudOfTextRectangle(list_tr)
	cloud.arrange_text()
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
	resolve_conflicts_tree1 = cloud1.treat_conflicts([])
	resolve_conflicts_tree2 = cloud2.treat_conflicts([])
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