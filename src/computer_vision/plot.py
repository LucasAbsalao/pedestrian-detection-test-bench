import matplotlib.pyplot as plt
import numpy as np

n_points = 400
t = np.arange(-20, 40, 60/n_points)

beta = 4
minimum_weight = 0.2
t_start = 10
t_end = 30

delta = (t - t_start) / (t_end-t_start) 
y1 = 1 + minimum_weight - 1 / (1 + np.exp(-beta * (delta - 1/2)))
y2 = 1 + minimum_weight - 1 / (1 + np.exp(-beta * (2*delta - 1)))

fig, ax = plt.subplots(1,2)
ax[0].plot(t,y1, color = 'red')
ax[0].vlines([t_start, t_end], 0, 1+minimum_weight, linestyle='dashed')
ax[0].grid()

ax[1].plot(t,y2, color = 'blue')
ax[1].vlines([t_start, t_end], 0, 1+minimum_weight, linestyle='dashed')
ax[1].grid()

plt.show()

plt.plot(t,y1, color = 'red')
plt.plot(t,y2, color = 'blue')
plt.vlines([t_start, t_end], 0, 1+minimum_weight, linestyle='dashed')
plt.grid()
plt.show()