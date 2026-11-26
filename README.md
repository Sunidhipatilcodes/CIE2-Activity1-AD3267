Activity 1: Multiple Linear Regression
Course: Regression and Classification Algorithms (RCA)
Department: Artificial Intelligence and Data Science (AI & DS)
Class: T.Y. B.Tech | Semester: V | Academic Year:
2026--27
Assessment: CIE 2 -- Activity 1 (5 Marks)
1. Activity overview
This activity uses Multiple Linear Regression to predict a continuous
numerical target from two or more numerical input features.

The notebook demonstrates manual least-squares calculation on a small
sample, model fitting with Scikit-learn, comparison of a single-feature
model with a multiple-feature model, coefficient interpretation, model
evaluation, residual analysis, multicollinearity checking using Variance
Inflation Factor (VIF), and prediction for a new example.
2. Dataset details
- Dataset name: Diabetes Dataset
- Source: Scikit-learn
- Official dataset documentation:
  https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
- Number of observations: 442
- Number of input features: 10 numerical features
- Target column: target
- Total columns: 11 (10 input features and 1 continuous target)
The target is a quantitative measure of disease progression one year
after baseline. It is a continuous regression target and does not
represent a diabetes diagnosis. The dataset is used for educational
regression analysis only and must not be used for medical decisions.
Input features
The ten numerical input features are:

age, sex, bmi, bp, s1, s2, s3, s4, s5, and s6.

The feature values supplied by Scikit-learn's Diabetes dataset loader
are already centred and scaled.
3. Repository contents
  File                                                    Description
  Activity1_Multiple_Linear_Regression_Diabetes.ipynb   Jupyter Notebook containing the
code, manual calculations,
explanations, visualizations,
evaluation metrics and predictions
README.md                                             Activity overview, dataset source,
workflow, results and execution
instructions
The notebook loads the dataset directly using Scikit-learn's
load_diabetes(as_frame=True) function. A separate CSV dataset file is
not required to run the notebook.
4. Tools and libraries used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Jupyter Notebook or Google Colab
5. Implementation workflow
Step 1: Import libraries
Import the required libraries for data manipulation, visualization,
Linear Regression, evaluation metrics and Variance Inflation Factor
(VIF). The notebook also displays the Python and library versions.
Step 2: Load and understand the dataset
Load the Diabetes dataset using Scikit-learn's
load_diabetes(as_frame=True) function. Display the first five records,
dataset shape, feature names, data types and dataset information. A data
dictionary describes the ten input features and the continuous target.
Step 3: Inspect and clean the data
Check missing values, duplicate records and summary statistics. Remove
exact duplicate rows if present and fill missing numerical values using
column medians if required.

In the executed notebook, the dataset contained 0 missing values and 0
duplicate rows. The dataset therefore remained 442 rows × 11
columns after cleaning.
Step 4: Explore feature relationships
Create scatter plots for each input feature against the target.
Calculate the correlation matrix and display it as a heatmap.

The strongest positive correlations with the target were: - bmi:
0.5865 - s5: 0.5659

Correlation indicates association and does not prove causation.
Step 5: Perform manual multiple linear regression
Use the first eight observations and two features, bmi and s5, to
demonstrate a two-feature regression model:
b_2(`\text{s5}`{=tex}) \]

Construct the design matrix with a column of ones for the intercept.
Calculate (X\^T X) and (X\^T y), then solve the least-squares problem
using `numpy.linalg.lstsq`.

The manual calculation gives:

  Parameter                       Value
  ------------------------- -----------
  Intercept ((b_0))            158.1817
  BMI coefficient ((b_1))     -750.3041
  S5 coefficient ((b_2))      1652.0523

The fitted equation for this eight-observation sample is:

\[ `\hat{y}`{=tex} = 158.1817 - 750.3041(`\text{bmi}`{=tex}) +
1652.0523(`\text{s5}`{=tex}) \]

For the first observation, the actual target was **151.00**, the
predicted value was **144.7789**, and the residual (actual − predicted)
was **6.2211**.

### Step 6: Verify the manual calculation

Fit Scikit-learn's `LinearRegression` model using the same eight
observations and the same two features. The manual and library
intercepts and coefficients matched to floating-point precision,
verifying the manual least-squares calculation.

### Step 7: Split the dataset into training and testing sets

Split the dataset into 80% training data and 20% testing data using
`random_state=42`.

-   **Training records:** 353
-   **Testing records:** 89
-   **Input features:** 10

### Step 8: Build the single-feature baseline model

Calculate feature-target correlations using the training data only.
Select the feature with the strongest absolute correlation with the
target and fit a single-feature Linear Regression model.

-   **Selected feature:** `bmi`
-   **Single-feature test R²:** 0.2334

### Step 9: Train the multiple-feature regression model

Fit a Scikit-learn `LinearRegression` model using all ten input
features. The model estimates each feature's coefficient while holding
the other included features constant.

The fitted model intercept was **151.3456**. The individual coefficients
and fitted equation are displayed in the notebook.

### Step 10: Compare actual and predicted values

Create a comparison table containing the actual test target values,
single-feature predictions and multiple-feature predictions.

### Step 11: Evaluate the models

Evaluate both models on the same held-out test set using:

-   **MAE:** Mean Absolute Error
-   **MSE:** Mean Squared Error
-   **RMSE:** Root Mean Squared Error
-   **R²:** Coefficient of determination
-   **Adjusted R²:** R² adjusted for the number of predictors

### Step 12: Perform residual analysis

Calculate residuals as actual target values minus predicted target
values. Generate: - Residuals versus predicted values plot - Histogram
of residuals - Actual versus predicted values plot with an
ideal-prediction reference line

These plots help inspect patterns in prediction errors and assess
whether further investigation of model assumptions may be needed.

### Step 13: Check multicollinearity using VIF

Calculate the Variance Inflation Factor (VIF) for each training feature.
The notebook uses general guidelines: values near 1 indicate little
linear redundancy, values above 5 may merit investigation, and values
above 10 are often considered high. These are guidelines, not strict
universal cut-offs.

### Step 14: Examine standardized coefficients

Standardize the features using a scaler fitted on training data only,
then fit a separate Linear Regression model. Display standardized
coefficients to compare feature contributions on a common scale.

### Step 15: Predict a new example

Create a new input record using the median value of each training
feature. Use the multiple-feature model to predict its continuous target
value and check whether each input feature lies within its training-data
range.

## 6. Results from the notebook

The following results were obtained by executing the notebook with its
stated code and settings (`test_size=0.20`, `random_state=42`).

### Single-feature vs multiple-feature model

  -----------------------------------------------------------------------
  Metric                   Single-feature model    Multiple-feature model
                                        (`bmi`)             (10 features)
  ------------------- ------------------------- -------------------------
  MAE                                   52.2600               **42.7941**

  MSE                                 4061.8259             **2900.1936**

  RMSE                                  63.7323               **53.8534**

  R²                                     0.2334                **0.4526**

  Adjusted R²                            0.2245                **0.3824**
  -----------------------------------------------------------------------

### Result interpretation

The multiple-feature model performed better than the single-feature
model on the held-out test set. Its test R² increased from **0.2334 to
0.4526**, while RMSE decreased from **63.7323 to 53.8534** and MAE
decreased from **52.2600 to 42.7941**. The multiple-feature model
achieved an adjusted R² of **0.3824**.

These results indicate that, for this train-test split, using all ten
features improved prediction performance compared with using BMI alone.

### Multicollinearity results

The VIF values calculated by the notebook were:

  Feature         VIF
  --------- ---------
  s1          55.2580
  s2          35.7654
  s3          14.2906
  s5          10.0776
  s4           9.3332
  bmi          1.5750
  s6           1.5020
  bp           1.4229
  sex          1.2688
  age          1.2176

The VIF results indicate substantial multicollinearity among some
serum-related features, particularly `s1`, `s2` and `s3`. This means
some predictors contain overlapping linear information and individual
coefficient estimates may be less stable. The notebook reports VIF
values but does not remove these features from the multiple-feature
model.

### New example prediction

The notebook creates a new example using the median of each training
feature. The predicted target value was **157.53**. All ten feature
values in this example were within their respective training-data
ranges.

This prediction is an estimate of the dataset's quantitative target, not
a medical diagnosis.

## 7. Conclusion and decision log

The Scikit-learn Diabetes dataset was used to predict a continuous
disease-progression measure with Multiple Linear Regression. A manual
two-feature least-squares calculation using `bmi` and `s5` was verified
against Scikit-learn. The single-feature model using `bmi` achieved a
test R² of 0.2334 and RMSE of 63.7323, while the multiple-feature model
using all ten features achieved a test R² of 0.4526, adjusted R² of
0.3824 and RMSE of 53.8534.

Therefore, the multiple-feature model performed better on this test
split. Residual plots were generated to inspect prediction errors, and
VIF analysis identified multicollinearity among several serum-related
features. The new example prediction was 157.53. These findings are
specific to this dataset and split; the model is intended for
educational purposes only.

### Decision log

-   **Dataset choice:** Used Scikit-learn's Diabetes dataset because it
    contains numerical input features and a continuous target.
-   **Manual feature choice:** Used `bmi` and `s5` to demonstrate a
    two-feature least-squares calculation on eight observations.
-   **Single-feature baseline:** Selected `bmi` because it had the
    strongest absolute feature-target correlation in the training data.
-   **Model comparison:** Compared the single-feature model against a
    multiple-feature model using all ten features on the same test set.
-   **Multicollinearity:** Calculated VIF and identified high values
    among some serum-related features; the features were retained in the
    model and the issue was documented.
-   **Evaluation:** Used MAE, MSE, RMSE, R² and adjusted R², along with
    residual plots, to assess model performance.
-   **Limitations:** Observational data, possible multicollinearity,
    linear-model assumptions and no clinical use.

## 8. How to run the notebook

### Using Google Colab

1.  Download the notebook from this repository.
2.  Open https://colab.research.google.com/.
3.  Select **File → Upload notebook**.
4.  Upload `Activity1_Multiple_Linear_Regression_Diabetes.ipynb`.
5.  Select **Runtime → Run all**.
6.  Wait for the code cells, tables and plots to finish executing.
7.  Save the executed notebook with its outputs.

### Running locally

Install the required libraries:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
```

Start Jupyter Notebook:

``` bash
jupyter notebook
```

Open the notebook and run all cells from top to bottom.

## 9. Limitations

-   The dataset is observational; correlations and regression
    coefficients do not establish causation.
-   Some input features have high VIF values, indicating
    multicollinearity and potentially less stable individual
    coefficients.
-   Linear Regression may not capture non-linear relationships or all
    factors affecting the target.
-   Test-set results are based on one train-test split and may vary with
    a different split.
-   Predictions are estimates and are not guaranteed outcomes.
-   This activity is for educational purposes only. The dataset and
    model must not be used for medical diagnosis, screening or treatment
    decisions.

## 10. References

1.  Scikit-learn -- Diabetes dataset:
    https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
2.  Scikit-learn -- `load_diabetes` documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
3.  Scikit-learn -- Linear Regression:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
4.  Statsmodels -- Variance Inflation Factor:
    https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html

## 11. GitHub submission

Create a separate **public GitHub repository** for Activity 1 with the
following repository name:

`CIE2-Activity1-AD3267`

Include the submission reference requested in the activity instructions:

[complexiota GitHub](https://github.com/complexiota)

Before submitting, confirm that the repository is public and that its
URL opens in an incognito/private browser window without requesting
access. Do not upload passwords, API keys, confidential information or
personal data.
