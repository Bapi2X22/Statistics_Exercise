import numpy as np
import matplotlib.pyplot as plt

xb_l=[]
nBins = 400
b_tot = 1000.
numExp = 10
np.random.seed(seed=1234567)        # fix random seed
# for i in range(numExp):
#     n = np.random.poisson(b_tot)       # first b only
#     r = np.random.uniform(0., 1., n)
#     xb = 0.5*np.log(1/(1-r))
#     xb_l.append(xb)
n = np.random.poisson(b_tot)       # first b only
r = np.random.uniform(0., 1., n)
xb = 0.5*np.log(1/(1-r))

plt.hist(xb,bins=20,density=True)
x=np.linspace(0,4,1000)
y=2*np.exp(-2*x)
plt.plot(x,y)
plt.show()





# print(xb_l)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# Hist, bin_edges = np.histogram(xb_l, bins=nBins, density=True)

# Make and analyse histograms of q for b and s+b hypotheses
# nBins = 400
# qMin = -200.
# qMax = 0.
# qbHist, bin_edges = np.histogram(qb, bins=nBins, range=(qMin,qMax), density=True)
# qsbHist, bin_edges = np.histogram(qsb, bins=nBins, range=(qMin,qMax), density=True)
# med_q_sb = np.median(qsb)
# print("median[q|s+b]   = {:.3f}".format(med_q_sb))

# # Plot histograms of q
# binLo, binHi = bin_edges[:-1], bin_edges[1:]
# xPlot = np.array([binLo, binHi]).T.flatten()
# ybPlot = np.array([qbHist, qbHist]).T.flatten()
# ysbPlot = np.array([qsbHist, qsbHist]).T.flatten()
# # xPlot = np.array(binLo)
# # ybPlot = np.array(qbHist)
# # ysbPlot = np.array(qsbHist)
# fig, ax = plt.subplots(1,1)
# plt.yscale("log")
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.gcf().subplots_adjust(left=0.15)
# plt.xlabel(r'$q$', labelpad=5)
# plt.ylabel(r'$f(q)$', labelpad=10)
# plt.plot(xPlot, ybPlot, label=r'$f(q|b)$', color='dodgerblue')
# plt.plot(xPlot, ysbPlot, label=r'$f(q|s+b)$', color='orange')
# ax.axvline(med_q_sb, color="black", linestyle="dashed", label = r'median$[q|s+b]$')
# ax.legend(loc='upper left', frameon=False)
# plt.show()

# plt.hist(xb_l,density=True)

# x=np.linspace(0,4,1000)
# y=2*np.exp(-2*x)
# plt.plot(x,y)
# plt.show()

