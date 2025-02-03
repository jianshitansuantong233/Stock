from copy import deepcopy

import matplotlib.pyplot as plt
import csv
import bisect
import numpy as np

class data:
	def __init__(self):
		self.eps = -1
		# Unit is M
		self.shares_outstanding = -1
		self.total_assets = -1
		self.total_liabilities = -1
		self.asset_minus_liability = -1
		self.net_income = -1
		self.sales = -1
		self.gross_profit = -1
		self.pe = -1
		self.price = -1
		self.market_cap = -1
		self.margin = -1


class company:
	def __init__(self, symbol=""):
		self.symbol = symbol
		self.ordered_dates = []
		self.data = {}
	def add(self, date, data):
		if date < "2016-12-31":
			return
		if date in self.data.keys():
			for i in vars(data):
				if getattr(data,i) != -1:
					setattr(self.data[date], i,  getattr(data,i))
		else:
			bisect.insort(self.ordered_dates, date)
			try:
				temp = self.ordered_dates.index(date)
				self.data[date] = deepcopy(self.data[self.ordered_dates[temp - 1]])
				for i in vars(data):
					if getattr(data, i) != -1:
						setattr(self.data[date], i, getattr(data, i))
			except:
				self.data[date] = data
		setattr(self.data[date],"asset_minus_liability", getattr(self.data[date],"total_assets")-getattr(self.data[date],"total_liabilities"))
		setattr(self.data[date], "pe", getattr(self.data[date], "price") / getattr(self.data[date], "eps"))
		setattr(self.data[date], "market_cap", getattr(self.data[date], "price") * getattr(self.data[date], "shares_outstanding"))
		setattr(self.data[date], "margin",getattr(self.data[date], "gross_profit") / getattr(self.data[date], "sales"))

def reject_outliers(data, m=2):
	mean = np.mean(data)
	std = np.std(data)
	removal_index = []
	for i in range(len(data)):
		if abs(data[i] - mean) > m * std:
			removal_index.append(i)
	return data[abs(data - np.mean(data)) < m * np.std(data)], removal_index


LRCX = company()

with open('temp1.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
	state = 0
	for row in lines:
		if "period_end_date" in row:
			print("eps")
			state = 1
		elif "shares_outstanding" in row:
			print("shares outstanding")
			state = 2
		elif "total_assets" in row:
			print("total assets")
			state = 3
		elif "total_liabilities" in row:
			print("total liabilities")
			state = 4
		elif "net_income" in row:
			print("net income")
			state = 5
		elif "sales" in row:
			print("sales")
			state = 6
		elif "gross_profit" in row:
			print("gross_profit")
			state = 7
		else:
			match state:
				case 1:
					new_data = data()
					setattr(new_data,"eps",float(row[1]))
					LRCX.add(row[0],new_data)
				case 2:
					new_data = data()
					setattr(new_data, "shares_outstanding", float(row[1])/(10^6))
					LRCX.add(row[0], new_data)
				case 3:
					new_data = data()
					setattr(new_data, "total_assets", float(row[1]))
					LRCX.add(row[0], new_data)
				case 4:
					new_data = data()
					setattr(new_data, "total_liabilities", float(row[1]))
					LRCX.add(row[0], new_data)
				case 5:
					new_data = data()
					setattr(new_data, "net_income", float(row[1]))
					LRCX.add(row[0], new_data)
				case 6:
					new_data = data()
					setattr(new_data, "sales", float(row[1]))
					LRCX.add(row[0], new_data)
				case 7:
					new_data = data()
					setattr(new_data, "gross_profit", float(row[1]))
					LRCX.add(row[0], new_data)


with open('temp.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
	state = 0
	for row in lines:
		if "close" in row:
			print("price")
			state = 1
		else:
			match state:
				case 1:
					new_data = data()
					setattr(new_data, "price", float(row[1]))
					LRCX.add(row[0], new_data)


market_cap = []
pe = []
shares_outstanding = []
book = []
price = []
margin = []
for i in LRCX.ordered_dates:
	market_cap.append(getattr(LRCX.data[i], "market_cap"))
	pe.append(getattr(LRCX.data[i], "pe"))
	#shares_outstanding.append(getattr(LRCX.data[i], "shares_outstanding"))
	book.append(getattr(LRCX.data[i], "asset_minus_liability"))
	#price.append(getattr(LRCX.data[i], "price"))
	margin.append(getattr(LRCX.data[i], "margin"))

#market_cap,removal_index = reject_outliers(np.array(market_cap))
#X = deepcopy(LRCX.ordered_dates)
#for i in removal_index:
#	X.pop(i)

#margin,removal_index = reject_outliers(np.array(margin))
X = deepcopy(LRCX.ordered_dates)
#for i in removal_index:
#	X.pop(i)

plt.plot(X, margin, color = 'g', linestyle = 'dashed',
		marker = 'o')

plt.xticks(rotation = 25)
plt.xlabel('Dates')
plt.ylabel('Dollar')
plt.title('LRCX Income Report', fontsize = 20)
plt.grid()
plt.legend()
plt.show()
