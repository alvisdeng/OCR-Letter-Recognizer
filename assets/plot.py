import matplotlib.pyplot as plt
import numpy as np

num_epoch = np.arange(1,101)
train_cross_entropy = []
validation_cross_entropy = []
with open('cross_entropy/LR3.txt',mode='r') as f:
    for line in f.readlines():
        splitted_line = line.strip().split(' ')
        tag = splitted_line[1]
        entropy = splitted_line[2]
        if tag == 'crossentropy(train)':
            train_cross_entropy.append(float(entropy))
        else:
            validation_cross_entropy.append(float(entropy))

plt.plot(num_epoch,train_cross_entropy, ls='--', marker='o', label='Train Cross Entropy',markersize=2)
plt.plot(num_epoch,validation_cross_entropy,  ls='-', marker='v', label='Validation Cross Entropy',markersize=2)
plt.xlabel('Num of Epoch')
plt.ylabel('Cross Entropy')
plt.title('Large Dataset Train/Validation Cross Entropy (LR=0.001)')
plt.legend()
plt.show()