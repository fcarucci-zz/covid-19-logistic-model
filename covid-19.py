import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import dateutil.parser as dp

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"

df = pd.read_csv(url)


deaths = df.loc[:,['data','deceduti']]
#deaths = df.loc[:,['data','terapia_intensiva']]

start_date = dp.parse('2020-01-01T00:00:00')

deaths['data'] = deaths['data'].map(lambda x : (dp.parse(x) - start_date).days)

print deaths

def logistic_model(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))

x = list(deaths.iloc[:, 0])
y = list(deaths.iloc[:, 1])

fit = curve_fit(logistic_model, x, y, p0 = [2, 100, 10000])

errors = [np.sqrt(fit[1][i][i]) for i in [0, 1, 2]]

print errors

print "speed = %f" % fit[0][0]
print "inflection = %d %s" % (fit[0][1], start_date + timedelta(days=fit[0][1]))
print "max = %d +- %d" % (fit[0][2], errors[2])

a = fit[0][0]
b = fit[0][1]
c = fit[0][2]
solution = int(fsolve(lambda x : logistic_model(x, a, b, c) - int(c), b))

a = fit[0][0] + errors[0]
b = fit[0][1] + errors[1]
c = fit[0][2] + errors[2]
solution = int(fsolve(lambda x : logistic_model(x, a, b, c) - int(c), b))

print "end  = %d %s" % (solution, start_date + timedelta(days=solution))


pred_x = list(range(max(x),solution))
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x])
plt.plot(x+pred_x, [logistic_model(i,fit[0][0] - errors[0],fit[0][1] - errors[1],fit[0][2] - errors[2]) for i in x+pred_x])
plt.plot(x+pred_x, [logistic_model(i,fit[0][0] + errors[0],fit[0][1] + errors[1],fit[0][2] + errors[2]) for i in x+pred_x])
# Predicted exponential curve
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of deaths")
plt.ylim((min(y)*0.9,c*1.1))
plt.show()
