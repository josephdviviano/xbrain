#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

a = np.genfromtxt('template_variances_IM.csv', skip_header=1, delimiter=',')
b = np.genfromtxt('template_variances_OB.csv', skip_header=1, delimiter=',')
c = np.genfromtxt('template_variances_REST.csv', skip_header=1, delimiter=',')

a = np.sum(a, axis=1)
b = np.sum(b, axis=1)
c = np.sum(c, axis=1)

plt.plot(a, color='green')
plt.plot(b, color='red')
plt.plot(c, '-', color='black')

