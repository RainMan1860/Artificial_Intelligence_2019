from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print(mean_absolute_error(y_true, y_pred))
print(mean_squared_error(y_true, y_pred))
print(pearsonr(y_true, y_pred))
print(spearmanr(y_true, y_pred))


y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(mean_absolute_error(y_true, y_pred))
print(mean_squared_error(y_true, y_pred))
