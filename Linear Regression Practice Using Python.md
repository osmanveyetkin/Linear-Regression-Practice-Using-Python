# Linear Regression Practice Using Python

This study involves practical application of regression analysis, as covered in the Google Advanced Data Analytics course.

For more information, you may refer to the official course materials provided by Coursera:  
[Google Advanced Data Analytics Certificate](https://www.coursera.org/professional-certificates/google-advanced-data-analytics)

## Call İmports



Begin by importing the relevant packages and data.


```python
# Import packages
import pandas as pd
import seaborn as sns
```

**Note:** Loading the dataset and inspecting the first few rows using the head() function


```python
# Load dataset
penguins = sns.load_dataset("penguins")

# Examine first 5 rows of dataset
penguins.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>



From the first 5 rows of the dataset, we can see that there are several columns available: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, and `sex`. There also appears to be some missing data.

## Data cleaning

From the first 5 rows of the dataset, we can see that there are several columns available: species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, and sex. There also appears to be some missing data. We know from the course materials that this dataset contains information about penguins and includes some missing values that need to be handled during data preprocessing.



```python
# Keep Adelie and Gentoo penguins, drop missing values
penguins_sub = penguins[penguins["species"] != "Chinstrap"]
penguins_final = penguins_sub.dropna()
penguins_final.reset_index(inplace=True, drop=True)
```

## Exploratory data analysis

Before you construct any model, it is important to get more familiar with your data. You can do so by performing exploratory data analysis or EDA. Please review previous program materials as needed if you would like to refamiliarize yourself with EDA concepts.

Since this part of the course focuses on simple linear regression, you want to check for any linear relationships among variables in the dataframe. You can do this by creating scatterplots using any data visualization package, for example `matplotlib.plt`, `seaborn`, or `plotly`.

To visualize more than one relationship at the same time, we use the `pairplot()` function from the `seaborn` package to create a scatterplot matrix.


```python
# Create pairwise scatterplots of data set
sns.pairplot(penguins_final)
```




    <seaborn.axisgrid.PairGrid at 0x1117fad80>




​    
![png](output_14_1.png)
​    


From the scatterplot matrix, you can observe a few linear relationships:
* bill length (mm) and flipper length (mm)
* bill length (mm) and body mass (g)
* flipper length (mm) and body mass (g)

## Model construction



Based on the above scatterplots, you could probably run a simple linear regression on any of the three relationships identified. For this part of the course, you will focus on the relationship between bill length (mm) and body mass (g).

To do this, you will first subset the variables of interest from the dataframe. You can do this by using double square brackets `[[]]`, and listing the names of the columns of interest.


```python
# Subset Data
ols_data = penguins_final[["bill_length_mm", "body_mass_g"]]
```

Next, you can construct the linear regression formula, and save it as a string. Remember that the y or dependent variable comes before the `~`, and the x or independent variables comes after the `~`.

**Note:** The names of the x and y variables have to exactly match the column names in the dataframe.


```python
# Write out formula
ols_formula = "body_mass_g ~ bill_length_mm"
```

Lastly, you can build the simple linear regression model in `statsmodels` using the `ols()` function. You can import the `ols()` function directly using the line of code below.


```python
# Import ols function
from statsmodels.formula.api import ols
```

Then, you can plug in the `ols_formula` and `ols_data` as arguments in the `ols()` function. After you save the results as a variable, you can call on the `fit()` function to actually fit the model to the data.


```python
# Build OLS, fit model to data
OLS = ols(formula = ols_formula, data = ols_data)
model = OLS.fit()
```

Lastly, you can call the `summary()` function on the `model` object to get the coefficients and more statistics about the model. The output from `model.summary()` can be used to evaluate the model and interpret the results. Later in this section, we will go over how to read the results of the model output.


```python
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>body_mass_g</td>   <th>  R-squared:         </th> <td>   0.769</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.768</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   874.3</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 07 May 2025</td> <th>  Prob (F-statistic):</th> <td>1.33e-85</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:38:08</td>     <th>  Log-Likelihood:    </th> <td> -1965.8</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   265</td>      <th>  AIC:               </th> <td>   3936.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   263</td>      <th>  BIC:               </th> <td>   3943.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>      <td>-1707.2919</td> <td>  205.640</td> <td>   -8.302</td> <td> 0.000</td> <td>-2112.202</td> <td>-1302.382</td>
</tr>
<tr>
  <th>bill_length_mm</th> <td>  141.1904</td> <td>    4.775</td> <td>   29.569</td> <td> 0.000</td> <td>  131.788</td> <td>  150.592</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 2.060</td> <th>  Durbin-Watson:     </th> <td>   2.067</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.357</td> <th>  Jarque-Bera (JB):  </th> <td>   2.103</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.210</td> <th>  Prob(JB):          </th> <td>   0.349</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.882</td> <th>  Cond. No.          </th> <td>    357.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



You can use the `regplot()` function from `seaborn` to visualize the regression line.


```python
sns.regplot(x = "bill_length_mm", y = "body_mass_g", data = ols_data)
```




    <Axes: xlabel='bill_length_mm', ylabel='body_mass_g'>




​    
![png](output_28_1.png)
​    


## Finish checking model assumptions


```python
# Subset X variable
X = ols_data["bill_length_mm"]

# Get predictions from model
fitted_values = model.predict(X)
```

Then, you can save the model residuals as a variable by using the `model.resid` attribute.


```python
# Calculate residuals
residuals = model.resid
```

### Check the normality assumption

To check the normality assumption, you can create a histogram of the residuals using the `histplot()` function from the `seaborn` package.

From the below histogram, you may notice that the residuals are almost normally distributed. In this case, it is likely close enough that the assumption is met.


```python
import matplotlib.pyplot as plt
fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()
```


​    
![png](output_35_0.png)
​    


Another way to check the normality function is to create a quantile-quantile or Q-Q plot. Recall that if the residuals are normally distributed, you would expect a straight diagonal line going from the bottom left to the upper right of the Q-Q plot. You can create a Q-Q plot by using the `qqplot` function from the `statsmodels.api` package.

The Q-Q plot shows a similar pattern to the histogram, where the residuals are mostly normally distributed, except at the ends of the distribution.


```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
fig = sm.qqplot(model.resid, line = 's')
plt.show()
```


​    
![png](output_37_0.png)
​    


### Check the homoscedasticity assumption

Lastly, we have to check the homoscedasticity assumption. To check the homoscedasticity assumption, you can create a scatterplot of the fitted values and residuals. If the plot resembles a random cloud (i.e., the residuals are scattered randomly), then the assumption is likely met.

You can create one scatterplot by using the `scatterplot()` function from the `seaborn` package. The first argument is the variable that goes on the x-axis. The second argument is the variable that goes on the y-axis.


```python
# Import matplotlib
import matplotlib.pyplot as plt
fig = sns.scatterplot(x=fitted_values, y=residuals)

# Add reference line at residuals = 0
fig.axhline(0)

# Set x-axis and y-axis labels
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")

# Show the plot
plt.show()
```


​    
![png](output_40_0.png)
​    

This project was developed alongside the Google Advanced Data Analytics course and focuses on practical implementation of linear regression. Leveraging the dataset and methodologies introduced in the course, it walks through data preprocessing, visualization, and model building using Python. The study complements the course content by reinforcing concepts through hands-on application.
