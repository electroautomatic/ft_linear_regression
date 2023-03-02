import matplotlib.pyplot as plt
import pandas as pd
from minimization_methods import gradientDescent


def getData(file):
	df = pd.read_csv(file)
	x = df['km'].values
	y = df['price'].values
	return x, y

def	normalizeData(mileages, prices):
	x = []
	y = []
	minM = min(mileages)
	maxM = max(mileages)
	for mileage in mileages:
		x.append((mileage - minM) / (maxM - minM))
	minP = min(prices)
	maxP = max(prices)
	for price in prices:
		y.append((price - minP) / (maxP - minP))
	return (x, y)

def	normalizeElem(list, elem):
	return ((elem - min(list)) / (max(list) - min(list)))

def	denormalizeElem(list, elem):
	return ((elem * (max(list) - min(list))) + min(list))

def denormalize_coef(k_norm, b_norm, x_orig, y_orig):
    x_min = min(x_orig)
    x_max = max(x_orig)
    y_min = min(y_orig)
    y_max = max(y_orig)
    x_range = x_max - x_min
    y_range = y_max - y_min

    k_denorm = k_norm * (y_range / x_range)
    b_denorm = - k_norm * x_min * (y_range / x_range) + b_norm * y_range + y_min
   
    return k_denorm, b_denorm

def	displayPlot(t0, t1, mileages, prices, lossHistory, t0History, t1History):
	lineX = [float(min(mileages)), float(max(mileages))]
	lineY = []
	for elem in lineX:
		elem = t1 * normalizeElem(mileages, elem) + t0
		lineY.append(denormalizeElem(prices, elem))
		
	plt.figure(1)
	plt.subplot(2, 3, 1)
	plt.plot(lossHistory, 'r.')
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.subplot(2, 3, 2)
	plt.plot(t0History, 'g.')
	plt.xlabel('iterations')
	plt.ylabel('t0')
	plt.subplot(2, 3, 3)
	plt.plot(t1History, 'b.')
	plt.xlabel('iterations')
	plt.ylabel('t1')
	plt.subplot(2, 1, 2)
	plt.plot(mileages, prices, 'bo', lineX, lineY, 'r-')
	plt.xlabel('mileage')
	plt.ylabel('price')
	plt.show()
	

def saveData(t0, t1, file):
	data = {'t0': [t0], 't1': [t1]}      
	df = pd.DataFrame(data)
	df.to_csv(file, index=False)

def	main():
	learningRate = 0.5
	iterations = 500
	
	mileages, prices = getData('data.csv')
	x, y = normalizeData(mileages, prices)	
	t0, t1, lossHistory, t0History, t1History = gradientDescent(x, y, learningRate, iterations)
	saveData(t0, t1, 'thetas.csv')
	displayPlot(t0, t1, mileages, prices, lossHistory, t0History, t1History)
	print("t0: {:.8}".format(t0))
	print("t1: {:.8}".format(t1))

	t1DeNorm, t0DeNorm = denormalize_coef(t1, t0, mileages, prices)
	saveData(t0DeNorm, t1DeNorm, 'thetas_DeNorm.csv')
	print("t0DeNorm: {:.8}".format(t0DeNorm))
	print("t1DeNorm: {:.8}".format(t1DeNorm))

if	__name__ == '__main__':
	main()