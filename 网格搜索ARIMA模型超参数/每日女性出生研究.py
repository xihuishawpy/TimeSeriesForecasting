import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# 评估给定订单的ARIMA模型（p，d，q）
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[:train_size], X[train_size:]
	history = list(train)
	# 进行预测
	predictions = []
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	return mean_squared_error(test, predictions)

# 评估ARIMA模型的p，d和q值的组合
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# load dataset
series = Series.from_csv('daily-total-female-births.csv', header=0)
# 评估参数
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(3)
q_values = range(3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)