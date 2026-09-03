import matplotlib.pyplot as plt
import numpy as np

n_points = 400
t = np.arange(-20, 40, 60/n_points)

beta = 4.0
minimum_weight = 0.2
t_start = 10
t_end = 30

delta = (t - t_start) / (t_end-t_start) 
y1 = 1 - 1 / (1 + np.exp(-7 * (2*delta - 1)))
y2 = 1 - 1 / (1 + np.exp(-beta * (2*delta - 1)))*(1-minimum_weight)

# --- First Figure: Side-by-Side ---
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(t, y1, color='red', label='Baseline Formulation')
ax[0].vlines([t_start, t_end], 0, 1+minimum_weight, linestyle='dashed')
ax[0].grid()
ax[0].legend() 

ax[1].plot(t, y2, color='blue', label='Proposed Formulation')
ax[1].vlines([t_start, t_end], 0, 1+minimum_weight, linestyle='dashed')
ax[1].grid()
ax[1].legend() 

plt.tight_layout()
plt.show()

# --- Second Figure: Combined ---
plt.figure(figsize=(8, 5))
plt.plot(t, y1, color='red', label='Baseline Formulation')
plt.plot(t, y2, color='blue', label='Proposed Formulation')
plt.vlines([t_start, t_end], 0, 1+minimum_weight, linestyle='dashed', color='gray')
plt.grid()
plt.legend() 
plt.show()