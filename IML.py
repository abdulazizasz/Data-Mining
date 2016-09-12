from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression



diabetes = load_diabetes()
boston = load_boston()



model = LinearRegression()

# model.fit(diabetes.data, diabetes.target)

# expected = diabetes.target
# predicted = model.predict(diabetes.data)


# print 'Linear Regression model \n Diabetes dataset'
# print 'Mean squared error = %0.3f '% mse(expected, predicted)
# print 'R2 score = %0.3f' % r2_score(expected, predicted)


model.fit(boston.data, boston.target)

expected = boston.target
predicted = model.predict(boston.data)

print 'Linear Regression model \n Boston'
print 'Mean squareed erro = %.3f ' % mse(expected,predicted)
print 'R2 score =%0.3f ' % r2_score(expected,predicted)