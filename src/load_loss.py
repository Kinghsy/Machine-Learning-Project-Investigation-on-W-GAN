import os
import json
import matplotlib.pyplot as plt
import numpy as np



file = os.path.join("..", "result", "gan", "loss.log")
with open(file, 'r') as f:
    s = f.readline()
s = s.strip()

log = json.loads(s)
print(log)


# print(len(log['lossG']))
# print(len(log['lossD']))
# plt.plot(np.array(log['lossG']), color="red")
# for i in range(30, 1000):
#     if log['lossD'][i] > -5:
#         log['lossD'][i] = 0.6*log['lossD'][i]+0.4*log['lossD'][i-1]
plt.plot(np.array(log['lossD']), color="blue")
plt.plot()
plt.xlabel('Training Epochs')
plt.ylabel('JSD Estimate')
plt.legend(['G: DCGAN w/ BN, D: DCGAN'])
# plt.ylim([-200000, -10])
plt.ylim([0, 1.6])
# plt.plot(np.array(log['lossD'])[150:], color='r')
# plt.plot(log['lossG'], color='g')
# plt.yscale('symlog')
plt.plot()

plt.show()

