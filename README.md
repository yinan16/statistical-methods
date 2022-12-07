# Statistical Methods for Data Science (2022)

Course material

## Table of contents

+ **Introduction**
+ **Lecture 1**: Data types and descriptive analysis
  + Statistical data types: categorical (nominal, ordinal), numerical (discrete, continuous)
  + Data containers: array (tensor), table
  + Descriptive analysis:
    + Descriptive statistics:
      + Centrality: sample mean, median, mode
      + Dispersion: standard deviation, range
      + Dependence: correlation, covariance
    + Visualization: bar chart, pie chart, histogram, scatter plot, heatmap, mosaic plot, etc
+ **Lecture 2**: Probability distribution
  + Experiment, sample space, event, random variable, PDF/PMF
  + Conditional probability
  + Example probability distributions:
    + Bernoulli distribution
    + Categorical distribution
    + Discrete uniform distribution
    + Gaussian distribution

+ **Lecture 3**: Mathematical modelling
  + Quantile, CDF
  + Q-Q plot
  + Mathematical modeling

+ **Lecture 4**: Parammeter (point) estimation
  + Steps of parameter estimation
  + Joint probability distribution, independent and identically distributed (i.i.d.) random variables
  + Likelihood function, Maximum Likelihood Estimation (MLE)
  + Prior, posterior
  + Maximum A Posteriori estimation (MAP)

+ **Lecture 5**: Naive Bayes classifiers
  + Classification
  + Bayes' rule for multinomial and Gaussian naive Bayes classifiers
  + Implementation of naive Bayes classifiers
  + Evaluate a classifier
    + Training, validation and testing
    + TP, TN, FP, FN, accuracy, precision, recall, specificity, F1 score


+ **Lecture 6**: Central limit theorem and interval estimation
  + Sample statistic, sampling distribution, sample mean, sample variance, standard normal distribution
  + Central limit theorem
  + Interval estimation: confidence interval, credible interval


+ **Lecture 7**: Clustering Part I
  + Clustering
  + Cluster tendency
  + Centroid clustering: K-means
  + SSE and Silhouette score

+ **Lecture 8**: Clustering Part II
  + Distribution clustering
    + Gaussian Mixture Models (GMM)
      + EM algorithm
      + AIC/BIC
    + GMM vs Gaussian naive Bayes
    + GMM vs K-means
    + EM vs K-means parameter estimation
  + Hierarchical clustering, density clustering
  + Cluster validation


+ **Lecture 9**: Hypothesis testing part I
  + Terminology:
    - Experiment and parameter of interest
    - Null hypothesis and alternative hypothesis
    - Test statistic
    - Null distribution
    - Significance level, power and p-value
  + Examples: one sample z-test
  + p-hacking

+ **Lecture 10**: Hypothesis testing part II
  + Test statistics and hypothesis tests
    + One-sample z-test, two-sample z-test
    + One-sample t-test, two-sample t-test (unequal variances)
    + Paired t-test
    + Binomial test
    + McNemar’s test
  + Compare two classifiers
    + K-fold cross validation: paired t-test
    + Training-validation split and leave-one-out cross validation: McNemar's test
