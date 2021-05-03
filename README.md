### adjust_matplot_text

Arrange annotation text in pyplot, in order to avoid as much as possible overlapping.

--> not designed to be efficient when enlarging the window/zooming


before :

![caption](/media/example_before.png)

after :

![caption](/media/example_after.png)

code of example :
```
from adjust_matplot_text import *

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
```
