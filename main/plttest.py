import matplotlib.pyplot as plt
import numpy as np

"""
n = 12
x = np.linspace(0, 2*np.pi, 100)
for i in range(n):
    plt.plot(x, np.sin(x+np.pi/n*i), label=i)

plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()
"""

def RandomInit(fig):
	if fig in [0, 1]:
		p = np.random.rand(4) * 1.5

	return p

c=0
while c<10:
    print(RandomInit(0))
    c=c+1