import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.backend import categorical_crossentropy


imgX = 16
imgY = 8


# Build an adversarial example
def adversarial(model, y_goal, steps, epsilon):
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    x = np.random.randint(2, size=(imgX, imgY)) # Create a random input to initialize gradient descent
    x = x.reshape(1, imgX, imgY, 1).astype('float32')
    x = tf.convert_to_tensor(x)
    for i in range(steps): # Gradient descent on the input
        loss = categorical_crossentropy(y_goal, model(x)) # model(x) is the output of the trained model given the input x
        grad, = tf.gradients(loss, x) # Gradient of the loss function with respect to the input x
        x = x - (epsilon * grad) # Update x: move in the opposite direction of the gradient with step size epsilon
    sess.run(x)
    return x.eval() # Return the built adversarial example


model = load_model('model.h5')
y_goal = np.zeros(26)
y_goal[0] = 1 # yGoal is the letter "a"
steps = 50
epsilon = 50

x = adversarial(model, y_goal, steps, epsilon)
prediction = model.predict(x)
print("Predicted letter: ", chr(np.argmax(prediction) + ord('a'))) # Convert the predicted class into a letter
print("Confidence: ", round(np.amax(prediction)*100, 2), "%") # Confidence of the model in its prediction
