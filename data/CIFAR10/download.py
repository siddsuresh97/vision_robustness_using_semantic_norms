import os
from keras.datasets import cifar10
import matplotlib.pyplot as plt

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Load CIFAR 10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Create directories for storing data
if not os.path.exists('train'):
    os.makedirs('train')

if not os.path.exists('val'):
    os.makedirs('val')

# Save the training data
for i in range(len(y_train)):
    class_name = str(classes[y_train[i][0]])
    dir_name = 'train/' + class_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    image_name = dir_name + '/' + str(i) + '.png'
    plt.imsave(image_name, x_train[i])
    
# Save the validation data
for i in range(len(y_test)):
    class_name = str(classes[y_test[i][0]])
    dir_name = 'val/' + class_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    image_name = dir_name + '/' + str(i) + '.png'
    plt.imsave(image_name, x_test[i])
