from copy import deepcopy

import matplotlib.pyplot as plt
import csv
import bisect
import numpy as np
import argparse

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
		try:
			setattr(self.data[date],"asset_minus_liability", getattr(self.data[date],"total_assets")-getattr(self.data[date],"total_liabilities"))
			setattr(self.data[date], "pe", getattr(self.data[date], "price") / getattr(self.data[date], "eps"))
			setattr(self.data[date], "market_cap", getattr(self.data[date], "price") * getattr(self.data[date], "shares_outstanding"))
			setattr(self.data[date], "margin",getattr(self.data[date], "gross_profit") / getattr(self.data[date], "sales"))
		except ArithmeticError:
			print("Error with data")

def reject_outliers(data, m=2):
	mean = np.mean(data)
	std = np.std(data)
	removal_index = []
	for i in range(len(data)):
		if abs(data[i] - mean) > m * std:
			removal_index.append(i)
	return data[abs(data - np.mean(data)) < m * np.std(data)], removal_index

def parse_earning(company):
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
						company.add(row[0],new_data)
					case 2:
						new_data = data()
						setattr(new_data, "shares_outstanding", float(row[1])/(10^6))
						company.add(row[0], new_data)
					case 3:
						new_data = data()
						setattr(new_data, "total_assets", float(row[1]))
						company.add(row[0], new_data)
					case 4:
						new_data = data()
						setattr(new_data, "total_liabilities", float(row[1]))
						company.add(row[0], new_data)
					case 5:
						new_data = data()
						setattr(new_data, "net_income", float(row[1]))
						company.add(row[0], new_data)
					case 6:
						new_data = data()
						setattr(new_data, "sales", float(row[1]))
						company.add(row[0], new_data)
					case 7:
						new_data = data()
						setattr(new_data, "gross_profit", float(row[1]))
						company.add(row[0], new_data)

def parse_stock(company):
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
						company.add(row[0], new_data)

def plot(company, x, y, poly1d_fn):

	plt.plot(x, y, 'yo', x, poly1d_fn(range(len(x))), '--k')
	#plt.plot(x, y, color='g', linestyle='dashed', marker='o', x, poly1d_fn(x))

	plt.xticks(rotation=25)
	plt.xlabel('Dates')
	plt.ylabel('Dollar')
	plt.title(company.symbol+'Income Report', fontsize=20)
	plt.grid()
	plt.legend()
	plt.show()

def judgement(company):
	market_cap = []
	pe = []
	book = []
	margin = []
	net_income = []
	score = 0
	for i in company.ordered_dates:
		market_cap.append(getattr(company.data[i], "market_cap"))
		pe.append(getattr(company.data[i], "pe"))
		book.append(getattr(company.data[i], "asset_minus_liability"))
		margin.append(getattr(company.data[i], "margin"))
		net_income.append(getattr(company.data[i], "net_income"))
	# Judgement 1: growing margin
	#margin, removal_index = reject_outliers(np.array(margin))
	#X = deepcopy(company.ordered_dates)
	#for i in removal_index:
	#	X.pop(i)
	growth_rate, base = np.polyfit(range(len(company.ordered_dates)),margin, 1)
	print("margin growth rate: "+str(growth_rate))
	print("margin base: "+str(base))
	if growth_rate > 0:
		score = score + 1
	# Judgement 2: growing income
	#net_income, removal_index = reject_outliers(np.array(net_income))
	#X = deepcopy(company.ordered_dates)
	#for i in removal_index:
	#	X.pop(i)
	growth_rate, base = np.polyfit(range(len(company.ordered_dates)), net_income, 1)
	print("income growth rate: "+str(growth_rate))
	print("income base: "+str(base))
	if growth_rate > 0:
		score = score + 1


	poly1d_fn = np.poly1d((growth_rate,base))
	#plot(company, X, margin, poly1d_fn)



	return score

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Add a string argument using add_argument()
	parser.add_argument("--name", type=str, help="ticker")

	# Parse the command-line arguments
	args = parser.parse_args()
	# Access the string options
	name = args.name
	print("Processing stock " + name)
	c = company(name)
	parse_earning(c)
	parse_stock(c)
	score = judgement(c)
	print(score)