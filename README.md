# Performing logistic regression, LDA, KNN and CART on Boston Dataset
## Dataset: 
Dataset used is Boston: Housing Values in Suburbs of Boston dataset in MASS library in R. 

#### Format: 
This data frame contains the following columns:

crim :
per capita crime rate by town.

zn :
proportion of residential land zoned for lots over 25,000 sq.ft.

indus :
proportion of non-retail business acres per town.

chas :
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

nox :
nitrogen oxides concentration (parts per 10 million).

rm :
average number of rooms per dwelling.

age :
proportion of owner-occupied units built prior to 1940.

dis :
weighted mean of distances to five Boston employment centres.

rad :
index of accessibility to radial highways.

tax :
full-value property-tax rate per \$10,000.

ptratio :
pupil-teacher ratio by town.

black :
\(1000(Bk - 0.63)^2\) where \(Bk\) is the proportion of blacks by town.

lstat :
lower status of the population (percent).

medv :
median value of owner-occupied homes in \$1000s.

## Libraries: 
-- ISLR <br/>
-- corrplot <br/>
-- MASS <br/>
-- klaR <br/>
-- leaps <br/>
-- lattice <br/>
-- ggplot2 <br/>
-- corrplot <br/>
-- car <br/>
-- caret <br/>
-- class <br/>
-- rpart <br/>

## Logistic Regression: 

Logistic Regression uses linear regression with the addition of sigmoid function which helps in returning output in between 0-1 range. Here as i have two categorical features 0
and 1, if the logistic regression returns value lower than 0.5 I’ll classify that as 0 and it returns value more than that then I’ll classify that as 1.

#### Result: 
Logistic Regression gives accuracy of 87.25% and Error Rate of 12.74%.

## LDA:
Linear Discriminant analysis is a true decision boundary discovery algorithm. It assumes that the class has common covariance and it’s decision boundary is linear separating the class.

#### Result: 
LDA gives accuracy of 86.27% and Error rate of 13.72%.

## KNN: 
At k = 5 , KNN gives accuracy of 89.2% and error rate of 10.7%.

## CART:
Classification for decision Trees A decision tree is a flowchart-like structure in which each internal node represents a “test” on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label. The paths from root to leaf represent classification rules.

#### Result:
Decision Tree gives accuracy of 97.05% and error rate of 2.94%.

## Comparing All Models: 
![model_results_ques2](https://user-images.githubusercontent.com/46763031/148008333-1d2d7ada-7d7d-4c8e-849a-592eb3d98d1f.png)


