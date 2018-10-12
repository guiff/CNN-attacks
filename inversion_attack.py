import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model


imgX = 16
imgY = 8


def inversion(model, j, steps, epsilon): # j = 0 for the letter a, 1 for b, etc.
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    cost_history = []
    x_history = []
    
    x = np.random.randint(2, size=(imgX, imgY)) # Initialize x with a random image
    x = x.reshape(1, imgX, imgY, 1).astype('float32')
    x = tf.convert_to_tensor(x)
    x_history.append(x)
    
    for i in range(steps): # Gradient descent
        x = x_history[i]
        cost = 1 - model(x)[0][j] # model(x)[0][j] is the prediction confidence of the model for the considered letter
        cost_history.append(cost.eval())
        grad, = tf.gradients(cost, x) # Gradient of the cost with respect to the input x
        x = x - (epsilon * grad) # Update x: move in the opposite direction of the gradient with step size epsilon
        x_history.append(x)

    min_cost = np.argmin(cost_history)
    return cost_history[min_cost], x_history[min_cost].eval()


def plot(x):
    fig, ax = plt.subplots()
    ax.imshow(x, cmap='GnBu', interpolation='nearest')
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ax.annotate(str(x[i, j])[0], xy=(j, i), horizontalalignment='center', verticalalignment='center')
    plt.show(block=False)


model = load_model('model.h5')
j = 0 # We consider the letter "a"
steps = 10
epsilon = 1000

cost, x = inversion(model, j, steps, epsilon)
print("Confidence: ", round((1 - cost)*100, 2), "%")
plot(x.reshape(imgX, imgY))
plt.show()
