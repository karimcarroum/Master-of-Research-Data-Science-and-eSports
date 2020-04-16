### Extending the Dataset to Include Interactions and Second Order Polynomial Terms
### Karim Carroum Sanz - karim.carroum01@estudiant.upf.edu

### Note: higher order interactions and/or terms can be introduced, but for the analysed features it is not useful.

import pandas as pd

data = pd.read_csv("sselected.csv")
data = data.iloc[:,1:(data.shape[1]+1)]

p = data.shape[1] - 1               # Original number of variables.
total_new_vars = p * (p + 1) / 2    # Number of variables after including interactions and second order polynomial terms.

column_count = 1
for variable in range(1, p + 1):
    for var in range(column_count + 1, 19):
        data["{}_{}".format(data.columns[variable], data.columns[var])] = data.iloc[:,variable] * data.iloc[:,var]
    column_count = column_count + 1
    data["{}2".format(data.columns[variable])] = data.iloc[:,variable]*2
print(data)

data.to_csv('sselected_complete.csv')
