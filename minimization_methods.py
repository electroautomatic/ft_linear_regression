
def	gradientDescent(mileages, prices, learningRate, iterations):
	lossHistory = []
	t0History = [0.0]
	t1History = [0.0]
	t0 = 0.0
	t1 = 0.0
	message = "max epoch reached"
	
	for iteration in range(iterations):
		dt0 = 0
		dt1 = 0
		for mileage, price in zip(mileages, prices):
			dt0 += (t1 * mileage + t0) - price
			dt1 += ((t1 * mileage + t0) - price) * mileage
		t0 -= dt0 / len(mileages) * learningRate
		t1 -= dt1 / len(prices) * learningRate
		loss = lossFunction(t0, t1, mileages, prices)
		if iteration % 10 == 0:
			print("epoch {} - loss: {:.8}".format(iteration, loss))
		t0, t1, learningRate = boldDriver(loss, lossHistory, t0, t1, dt0, dt1, learningRate, len(mileages))
		lossHistory.append(loss)
		t0History.append(t0)
		t1History.append(t1)
		if earlyStopping(lossHistory):
			message = "early stopped"
			break
	print("\nend: {}.".format(message))
	print("epoch {} - loss: {:.8}".format(iteration, loss))
	return (t0, t1, lossHistory, t0History, t1History)

# mean squared error (MSE)
def	lossFunction(t0, t1, mileages, prices): 
	loss = 0.0
	for mileage, price in zip(mileages, prices):
		loss += (price - (t1 * mileage + t0)) ** 2
	return (loss / len(mileages))

# check loss and adjust learning rate (reduces the number of epochs of training by 3 times)
def	boldDriver(loss, lossHistory, t0, t1, dt0, dt1, learningRate, length):
	newLearningRate = learningRate
	if len(lossHistory) > 1:
		if loss >= lossHistory[-1]:
			t0 += dt0 / length * learningRate
			t1 += dt1 / length * learningRate
			newLearningRate *=  0.5
		else:
			newLearningRate *= 1.05
	return (t0, t1, newLearningRate)

# check if the last 8 losses are equal 
def	earlyStopping(lossHistory):
	check = 8
	if len(lossHistory) > check:
		mean = sum(lossHistory[-(check):]) / check
		last = lossHistory[-1]
		if round(mean, 9) == round(last, 9): 
			return True
	return False