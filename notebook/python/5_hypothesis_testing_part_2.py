#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ----------------------------------------------------------------------
# Created: 2020-07-23
# Last-Updated: 
# Filename: hypothesis_testing.ipynb
# Author: Yinan Yu
# Description:  
# If you have any questions or comments, email yinan@chalmers.se or 
# yinan.yu@asymptotic.ai
# Note: the content of this file is subject to change
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import os
plt.style.use('seaborn-darkgrid')

fig_dir = "../lectures/figs/"
data_dir = "../data/"


# # Hypothesis testing

# ## Test statistics

# In[2]:


from statsmodels.stats import contingency_tables


# In[3]:


fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=150) 
xmin = -5
xmax = 5
x = np.linspace(-5, 5, num=1000)
pdf = stats.norm.pdf(x, 0, 1)

q0 = -1.8
q1 = -q0
ax.fill_between(x[(x<=q0)], 
                    pdf[(x<=q0)], 
                    color="tomato", alpha=1, 
                label="p-value")
ax.fill_between(x[(x>=q1)], 
                    pdf[(x>=q1)], 
                    color="tomato", alpha=1)

ax.plot(x, pdf)
ax.scatter(q1, 0.005, s=6, c="blue")
ax.annotate("Observation $z_0$", (q1, 0), 
            xytext=(10, 50), 
            textcoords="offset points", 
            color="k",
            arrowprops={"arrowstyle":"->", "color":"blue"}, 
                 fontsize=12, 
            bbox=dict(edgecolor="blue", facecolor="white"))
ax.set_title("z-test", fontsize=12)
ax.set_xlabel("z", fontsize=12)
ax.set_xlim(xmin, xmax)
ax.set_ylabel(r"$f(z\mid H_0)$", fontsize=12)
ax.legend()
ax.set_ylim(0);


# In[4]:


fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=150) 
xmin = -5
xmax = 5
x = np.linspace(-5, 5, num=1000)


N = 20
pdf = stats.t.pdf(x, N-1)
q0 = -1.8
q1 = -q0
ax.fill_between(x[(x<=q0)], 
                pdf[(x<=q0)], 
                color="tomato", alpha=1, 
                label="p-value")
ax.fill_between(x[(x>=q1)], 
                pdf[(x>=q1)], 
                color="tomato", alpha=1)

ax.plot(x, pdf)
ax.scatter(q1, 0.005, s=6, c="blue")
ax.annotate("Observation $t_0$", (q1, 0), 
            xytext=(10, 50), 
            textcoords="offset points", 
            color="k",
            arrowprops={"arrowstyle":"->", "color":"blue"}, 
                 fontsize=12, 
            bbox=dict(edgecolor="blue", facecolor="white"))
ax.set_title("t-test with df=N-1=%i"%(N-1), fontsize=12)
ax.set_xlabel("t", fontsize=12)
ax.set_xlim(xmin, xmax)
ax.set_ylabel(r"$f(t\mid H_0)$", fontsize=12)
ax.legend()
ax.set_ylim(0);


# In[9]:


# Binomial test
fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=150) 
s = 3 # left 
# s = 8 # right
n = 50
p = 0.1

x = np.asarray(list(range(0, n+1)))
pmf = stats.binom.pmf(x, n, p)
ax.bar(x, pmf, width=.2)
ax.set_xlim(0, n)

if s < n*p:
    pvalue = stats.binom.cdf(s, n=n, p=p)
    pvalue_stats = stats.binom_test(s, n=n, p=p, alternative="less")
    pmf_test_statistic = pmf[x<=s]
    x_test_statistic = x[x<=s]
    ax.bar(x_test_statistic, pmf_test_statistic, width=.2, color="tomato", label="one-tailed p-value")

    ax.annotate("Observation $k_0$", (s, 0), 
                xytext=(10, 50), 
                textcoords="offset points", 
                color="k",
                arrowprops={"arrowstyle":"->", "color":"blue"}, 
                     fontsize=12, 
                bbox=dict(edgecolor="blue", facecolor="white"))
    ax.scatter(s, 0.001, s=6, c="blue")
else:    
    pvalue = (1 - stats.binom.cdf(s, n=n, p=p) + stats.binom.pmf(s, n=n, p=p))
    pvalue_stats = stats.binom_test(s, n=n, p=p, alternative="greater")
    pmf_test_statistic = pmf[x>=s]
    x_test_statistic = x[x>=s]
    ax.bar(x_test_statistic, pmf_test_statistic, width=.2, color="tomato", label="one-tailed p-value")

    ax.annotate("Observation $k_0$", (s, 0), 
                xytext=(10, 50), 
                textcoords="offset points", 
                color="k",
                arrowprops={"arrowstyle":"->", "color":"blue"}, 
                fontsize=12, 
                bbox=dict(edgecolor="blue", facecolor="white"))
    ax.scatter(s, 0.001, s=6, c="blue") 
ax.set_title("(exact) Binomial test", fontsize=12)
ax.set_xlabel("k", fontsize=12)
ax.set_ylabel(r"$f(k\mid H_0)$", fontsize=12)    
ax.legend()

fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=150) 
xmin = -5
xmax = 5
x = np.linspace(-5, 5, num=1000)
pdf = stats.norm.pdf(x, 0, 1)

if s < p*n:
    q0 = (s-p*n)/(math.sqrt(p*n*(1-p)))
    q1 = -q0    
    ax.fill_between(x[(x<=q0)], 
                    pdf[(x<=q0)], 
                    color="tomato", alpha=1, 
                    label="one-tailed\np-value")
    ax.annotate("Observation $z_0$", (q0, 0), 
                xytext=(10, 50), 
                textcoords="offset points", 
                color="k",
                arrowprops={"arrowstyle":"->", "color":"blue"}, 
                     fontsize=12, 
                bbox=dict(edgecolor="blue", facecolor="white"))
    ax.scatter(q0, 0.005, s=6, c="blue")
else:
    q1 = (s-p*n)/(math.sqrt(p*n*(1-p)))
    q0 = -q1
    ax.fill_between(x[(x>=q1)], 
                    pdf[(x>=q1)], 
                    color="tomato", alpha=1, label="one-tailed\np-value")
    ax.annotate("Observation $z_0$", (q1, 0), 
                xytext=(10, 50), 
                textcoords="offset points", 
                color="k",
                arrowprops={"arrowstyle":"->", "color":"blue"}, 
                     fontsize=12, 
                bbox=dict(edgecolor="blue", facecolor="white"))
    ax.scatter(q1, 0.005, s=6, c="blue")
    
p_approx = stats.norm.cdf(q0, 0, 1)    



ax.plot(x, pdf)


ax.set_title("(approximate) Binomial test", fontsize=12)
ax.set_xlabel("z", fontsize=12)
ax.set_ylabel(r"$f(z\mid H_0)$", fontsize=12)
ax.set_xlim(xmin, xmax)
ax.legend(loc="upper right")
ax.set_ylim(0)

print("N, p", n, p)
print("p-value from stats.binom_test:", pvalue_stats)
print("Our exact binom test:", pvalue)
print("Our approximate binom test", p_approx) 


# In[6]:


# Exact symmetic two-tailed binomial test
p = 0.5 # p has to be 0.5 for the null distribution to be symmetric
fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=150) 
s = 22
n = 50

x = np.asarray(list(range(0, n+1)))
pmf = stats.binom.pmf(x, n, p)
ax.bar(x, pmf, width=.2)
ax.set_xlim(0, n)

pvalue_stats = stats.binom_test(s, n=n, p=p, alternative="two-sided")

pvalue = 2*min(stats.binom.cdf(s, n=n, p=p), 
               (1 - stats.binom.cdf(s, n=n, p=p) + stats.binom.pmf(s, n=n, p=p)))
if s < n*p:    
    pmf_test_statistic = pmf[x<=s]
    x_test_statistic = x[x<=s]
    ax.bar(x_test_statistic, pmf_test_statistic, width=.2, color="tomato", 
           label="1/2 p-value")

    ax.annotate("Observation $k_0$", (s, 0), 
                xytext=(30, 50), 
                textcoords="offset points", 
                color="k",
                arrowprops={"arrowstyle":"->", "color":"blue"}, 
                     fontsize=12, 
                bbox=dict(edgecolor="blue", facecolor="white"))
    ax.scatter(s, 0.001, s=6, c="blue")
else:    
    pmf_test_statistic = pmf[x>=s]
    x_test_statistic = x[x>=s]
    ax.bar(x_test_statistic, pmf_test_statistic, width=.2, color="tomato", 
           label="1/2 p-value")

    ax.annotate("Observation $k_0$", (s, 0), 
                xytext=(30, 50), 
                textcoords="offset points", 
                color="k",
                arrowprops={"arrowstyle":"->", "color":"blue"}, 
                fontsize=12, 
                bbox=dict(edgecolor="blue", facecolor="white"))
    ax.scatter(s, 0.001, s=6, c="blue") 
ax.set_title("(exact) two-tailed Binomial test with $p=0.5$", fontsize=12)
ax.set_xlabel("k", fontsize=12)
ax.set_ylabel(r"$f(k\mid H_0)$", fontsize=12)    
ax.legend()

print("N, p", n, p)
print("p-value from stats.binom_test:", pvalue_stats)
print("Our exact binom test:", pvalue)


# In[7]:


# McNemar's test for small n10 or n01

n10 = 11
n01 = 1
n00 = 90
n11 = 102
N = n10 + n01
table = [[n00, n10], [n01, n11]]

s = min(n10, n01)
p = 2*stats.binom.cdf(s, N, 0.5)
print("-- Our exact binomial test:")
print(" pvalue     ", p)
print("statistic  ", s)
mcnemar_result = contingency_tables.mcnemar(table, exact=True, correction=True)
print("-- Mcnemar's exact test using statsmodels.stats:\n", mcnemar_result)

fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=150) 
xmin = 0
xmax = 15
x = list(range(0, N+1))
pmf = stats.binom.pmf(x, N, 0.5)
ax.bar(x, pmf, width=.2)

x = np.asarray(x)
pmf_test_statistic = pmf[x<=s]
x_test_statistic = x[x<=s]
ax.bar(x_test_statistic, pmf_test_statistic, width=.2, color="tomato", label="1/2 p-value")

ax.legend(loc="upper left")
ax.set_xlabel("s")
ax.set_ylabel(r"$f(s\mid H_0)$")
ax.set_title("(exact) McNemar's test");


# In[15]:


# McNemar's test for large n10 and n01

n10 = 22
n01 = 15
n00 = 90
n11 = 102
table_large = [[n00, n10], [n01, n11]]
N = n01 + n10
print(n01+n10+n00+n11)
s = np.power(abs(n10-n01)-1, 2)/(N)
p = 1-stats.chi2.cdf(s, df=1)
print("-- Our approximate McNemar's test:")
print(" pvalue     ", p)
print("statistic  ", s)

mcnemar_approx = contingency_tables.mcnemar(table_large, 
                                            exact=False, 
                                            correction=True)
print("-- Mcnemar's test using statsmodels.stats:\n", mcnemar_approx)

fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=150) 
xmin = 0
xmax = 4
x = np.linspace(xmin, xmax, 1000)
pdf_chi2 = stats.chi2.pdf(x, df=1)
ax.plot(x, pdf_chi2)
ax.scatter(s, 0.005, s=6, c="blue", zorder=2)
ax.annotate("Observation $s_0$", (s, 0), 
            xytext=(10, 50), 
            textcoords="offset points", 
            color="k",
            arrowprops={"arrowstyle":"->", "color":"blue"}, 
                 fontsize=12, 
            bbox=dict(edgecolor="blue", facecolor="white"))
ax.fill_between(x[x>=s], pdf_chi2[x>=s], 
                color="tomato", label="p-value", zorder=1)
ax.set_xlim(0)
ax.set_ylim(0)
ax.set_xlabel("s")
ax.set_ylabel(r"$f(s\mid H_0)$")
ax.set_title("(approximate) McNemar's test")
ax.legend()

fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=150) 
xmin = 0
xmax = 15
x = list(range(0, N+1))
pmf = stats.binom.pmf(x, N, 0.5)
ax.bar(x, pmf, width=.2)

s = min(n10, n01)
p = 2*stats.binom.cdf(s, N, 0.5)
print("-- Our exact binomial test:")
print(" pvalue     ", p)
print("statistic  ", s)
mcnemar_exact = contingency_tables.mcnemar(table_large, 
                                           exact=True, 
                                           correction=True)
print("-- Exact Mcnemar's test using statsmodels.stats:\n", 
      mcnemar_exact)

s_exact = min(n10, n01)
x = np.asarray(x)
pmf_test_statistic = pmf[x<=s_exact]
x_test_statistic = x[x<=s_exact]
ax.bar(x_test_statistic, 
       pmf_test_statistic, 
       width=.2, 
       color="tomato", 
       label="1/2 p-value")
ax.scatter(s_exact, 0.001, s=6, c="blue")
ax.annotate("Observation $s_0$", (s_exact, 0), 
            xytext=(50, 50), 
            textcoords="offset points", 
            color="k",
            arrowprops={"arrowstyle":"->", "color":"blue"}, 
                 fontsize=12, 
            bbox=dict(edgecolor="blue", facecolor="white"))
ax.legend(loc="upper left")
ax.set_title("(exact) McNemar's test")
ax.set_xlim(0)
ax.set_ylim(0)
ax.set_xlabel("s")
ax.set_ylabel(r"$f(s\mid H_0)$");

