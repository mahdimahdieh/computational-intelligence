import numpy as np
			
def perceptron_train(P, t, lr=1, max_epochs=1000):
	w = np.zeros(2)
	b = 0.0
	epc = 0
	for epoch in range(max_epochs):
		errors = 0
		for x, target in zip(P, t):
			y = 1 if np.dot(w, x) + b > 0 else 0
			if y != target:
				errors += 1
				update = lr * (target - y)
				w += update * x
				b += update
		if errors == 0:
			break
		epc = epoch
	return w, b, epc+1
			
epsilons = [1, 2, 6]
for eps in epsilons:
	P = [
		np.array([0,1]), np.array([1,-1]), np.array([-1,1]),
		np.array([1,eps]), np.array([1,0]), np.array([0,0])
	]
	t = [0,1,1,0,0,1]
	w, b, epochs = perceptron_train(P, t)
	print(f"ε={eps}: w={w}, b={b}, epochs={epochs}")

