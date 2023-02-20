import numpy as np

success = []
for i in ['500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000', '5500', '6000', '6500', '7000',
          '7500', '8000']:
    data = np.loadtxt(i + '_t.txt')
    success.append(np.sum(data < 1000))
print(success)
