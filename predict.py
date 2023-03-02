import pandas as pd
import numpy as np

def	getTheta(file):
	df = pd.read_csv(file)
	t0 = df.iloc[0, 0]
	t1 = df.iloc[0, 1]
	return t0, t1

def	normalizeElem(list, elem):
	return ((elem - min(list)) / (max(list) - min(list)))

def	denormalizeElem(list, elem):
	return ((elem * (max(list) - min(list))) + min(list))

def	estimatePrice(thetas, mileage, mileages, prices):
	price = thetas[1] * normalizeElem(mileages, mileage) + thetas[0]
	return (denormalizeElem(prices, price))

def	main():
	try:
		t0, t1 = getTheta('thetas.csv')
		t0_d, t1_d =getTheta('thetas_DeNorm.csv')
		df = pd.read_csv('data.csv')
	except:
		print('Error: thetas.csv or data.csv not found')
		return
	mileages = df['km'].values
	prices = df['price'].values
	print('t0: ', t0)
	print('t1: ', t1)
	print('t0_d: ', t0_d)
	print('t1_d: ', t1_d)
	
	try:
		mileage = float(input('Enter mileage: '))
		if mileage < 0:
			print('Error: mileage must be positive!')
			return
	except:
		print('Error: mileage must be a number!')
		return
	price = estimatePrice((t0, t1), mileage, mileages, prices)
	print('Price: {:.2f}'.format(price))
	price_d = t0_d + t1_d * mileage
	print('Price denormalized: {:.2f}'.format(price_d))

if __name__ == "__main__":
	main()