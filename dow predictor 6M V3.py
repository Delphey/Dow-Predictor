import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing

#import data
dataset = np.genfromtxt('Dow Predictor V3.csv', delimiter=';')
##, names="Date, Truck, Dow+3, Housing, ManuUnem, Permits"

# remove outliers in factors
##dataset = preprocessing.robust_scale(dataset)

# define factors and searched value
factors = dataset[:, [0, 1, 2]]
quality = dataset[:, 3]


# make a version that is 0 to 1 scale
xnormal = preprocessing.MinMaxScaler().fit_transform(factors)

# add intercept to both original and normalized types
xnormal = sm.add_constant(xnormal)
x = sm.add_constant(factors)

# do OLS
result = sm.OLS(endog=quality, exog=x).fit().summary()
resultnormal = sm.OLS(endog=quality, exog=xnormal).fit().summary()

# output
print(result)
print(resultnormal)
