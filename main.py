import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma


if __name__ == '__main__':
    data = pd.read_csv("Salary_Data.csv")
    data.head()

    x_val = data[["YearsExperience"]]
    y_val = data[["Salary"]]

    # Compute Pearson correlation coefficient
    pearson_r, p_value = pearsonr(x_val, y_val)
    print("Pearson correlation coefficient =", pearson_r)

    model = LinearRegression()
    model.fit(x_val, y_val)
    prediction = model.predict(x_val)
    prediction_train = model.predict(x_val)

    # Plot
    plt.title("Cool Title")
    plt.xlabel("Years Of Experience")
    plt.ylabel("Salary")
    plt.plot(x_val, prediction_train, color="blue", label="Regression")
    plt.scatter(x_val, y_val, color="red", label="Data")

    plt.legend()
    plt.show()

    # Get Value
    experience_year = [[3], [2.5], [10]]
    predicted_salary = model.predict(experience_year)
    print(f"Salary 3 years: {predicted_salary[0]}, 2.5 years: {predicted_salary[1]}, 10 years: {predicted_salary[2]}")

    # Evaluate
    mse = mean_squared_error(y_val, prediction)
    r2 = r2_score(y_val, prediction)
    print(f"Mean squared error: {mse}")
    print(f"R-squared error: {r2}")

    regressor = LinearRegression()
    regressor.fit(x_val, y_val)
    # Get the coefficients
    slope = regressor.coef_[0][0]
    intercept = regressor.intercept_[0]
    print("Slope =", slope)
    print("Intercept =", intercept)

    # Statsmodels
    # Fit OLS model using statsmodels.api
    x_val_with_const = sma.add_constant(x_val.values)  # Add constant for intercept
    smModel = sm.ols(y_val.values, x_val_with_const).fit()
    #latest idea
    #smModel = sm.ols(formula='x ~ y', y_val.values, x_val_with_const).fit()
    slope_statsmodels = smModel.params[1]
    intercept_statsmodels = smModel.params[0]

    print("Slope_statsmodels = " + str(slope_statsmodels))
    print("Intercept_statsmodels = " + str(intercept_statsmodels))
    #plot fehlt
