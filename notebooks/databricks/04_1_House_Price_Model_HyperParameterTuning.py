# Databricks notebook source
# MAGIC %md
# MAGIC Perform imports

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col#, countDistinct, count, when, isnan
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from pyspark.ml import Pipeline
import numpy as np

#temporary hide error reports
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC Read data

# COMMAND ----------

model_data_df = spark.read.parquet("/FileStore/final_data")

# COMMAND ----------

model_data_df.printSchema()

# COMMAND ----------

model_data_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Train and test datasets

# COMMAND ----------

train_data_df, test_data_df = model_data_df.randomSplit([0.8, 0.2])

print(f"train_data_df count: {train_data_df.count()}\ntest_data_df count: {test_data_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ----
# MAGIC ## Model selection

# COMMAND ----------

# MAGIC %md
# MAGIC An important task in ML is model selection, or using data to find the best model or parameters for a given task. This is also called tuning.
# MAGIC 
# MAGIC Basically, I need to find the answers for following questions:
# MAGIC 
# MAGIC **I. Which technique** should I choose (linear regression, decision tree, random forest etc.)?
# MAGIC 
# MAGIC **II. Which value of hyperparameters** should I choose, e.g. if the decision tree with depth 3 is better than the decision tree with depth 5?
# MAGIC 
# MAGIC **III. Which metric** should I use for comparison of all models?

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC MLlib supports model selection using tools such as CrossValidator and TrainValidationSplit. These tools require the following items:
# MAGIC 
# MAGIC **I. Estimator**: algorithm or Pipeline to tune
# MAGIC 
# MAGIC **II. Set of ParamMaps**: parameters to choose from, sometimes called a “parameter grid” to search over
# MAGIC 
# MAGIC **III. Evaluator**: metric to measure how well a fitted Model does on held-out test data

# COMMAND ----------

# MAGIC %md
# MAGIC At a high level, these model selection tools work as follows:
# MAGIC 
# MAGIC - They split the input data into separate training and test datasets.
# MAGIC - For each (training, test) pair, they iterate through the set of ParamMaps:
# MAGIC     - For each ParamMap, they fit the Estimator using those parameters, get the fitted Model, and evaluate the Model’s performance using the Evaluator.
# MAGIC - They select the Model produced by the best-performing set of parameters.

# COMMAND ----------

# MAGIC %md
# MAGIC ### I. Estimator
# MAGIC 
# MAGIC **Linear Regression**
# MAGIC 
# MAGIC In our case we choose LinearRegression as Estimator

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol="features", labelCol="label", predictionCol="lrPrediction")

lrPipeline = Pipeline(stages = [lr])

# COMMAND ----------

# MAGIC %md
# MAGIC **Logistic Regression**
# MAGIC 
# MAGIC The same piece of code for LogisticRegression will look like:
# MAGIC 
# MAGIC `from pyspark.ml.classification import LogisticRegression`
# MAGIC 
# MAGIC `blr = LogisticRegression(featuresCol="features", labelCol="binLabel", predictionCol="blrPrediction")`

# COMMAND ----------

# MAGIC %md
# MAGIC ### II. ParamGRidBuilder
# MAGIC 
# MAGIC **Linear Regression**
# MAGIC 
# MAGIC All parametrs for Linear regression could be found at the documentation:

# COMMAND ----------

# MAGIC %md
# MAGIC https://spark.apache.org/docs/2.4.4/ml-classification-regression.html#regression
# MAGIC 
# MAGIC https://spark.apache.org/docs/2.4.0/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression
# MAGIC 
# MAGIC https://spark.apache.org/docs/2.4.4/ml-classification-regression.html#survival-regression

# COMMAND ----------

# MAGIC %md
# MAGIC But mains parametrs for **Linear Regression** function are following:
# MAGIC * **loss**: The loss function to be optimized. Supported options: squaredError, huber. (default: squaredError)
# MAGIC * **elasticNetParam**: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty. (default: 0.0)
# MAGIC * **regParam**: regularization parameter (>= 0). (default: 0.0)
# MAGIC * **maxIter**: max number of iterations (>= 0). (default: 100, current: 50)
# MAGIC * **tol**: the convergence tolerance for iterative algorithms (>= 0). (default: 1e-06)

# COMMAND ----------

# MAGIC %md
# MAGIC Define parameter grid for linear regression

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

lrParamGrid = ParamGridBuilder() \
                .addGrid(lr.regParam, [0.0, 0.1, 0.5, 1.0]) \
                .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.5])\
                .addGrid(lr.maxIter, [5, 15, 50, 100])\
                .build()

# COMMAND ----------

# MAGIC %md
# MAGIC L(w;x,y):=1/2*(wTx−y)^2 + α(λ∥w∥1)+(1−α)(λ/2∥w∥2^2),α∈[0,1],λ≥0

# COMMAND ----------

# MAGIC %md
# MAGIC **Logistic Regression**
# MAGIC 
# MAGIC Main parameters for **Logistic Regression** are nearly the same as in case of Linear Regression.
# MAGIC There is a couple interesting parameter in advance:
# MAGIC - threshold - threshold in binary classification prediction (range [0, 1]) has impact on predicted value. Default is 0.5
# MAGIC - probabilityCol - conditional probability, 1/(1 + exp(-rawPrediction_k))
# MAGIC - rawPredictionCol - Raw prediction for each possible label

# COMMAND ----------

# MAGIC %md
# MAGIC ### III. Evaluator
# MAGIC 
# MAGIC The Evaluator can be a RegressionEvaluator for regression problems, a BinaryClassificationEvaluator for binary data, or a MulticlassClassificationEvaluator for multiclass problems. 
# MAGIC 
# MAGIC **Linear Regression**
# MAGIC RegressionEvaluator options:
# MAGIC - **rmse** - root mean squared error (default)
# MAGIC - **mse** - mean squared error
# MAGIC - **r2** - r^2 metric
# MAGIC - **mae** - mean absolute error
# MAGIC 
# MAGIC In our case we are satisfied with the existing options of RegressionEvaluator and choose RMSE.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

lrEvaluator = RegressionEvaluator()\
                .setPredictionCol("lrPrediction")\
                .setMetricName("rmse")

# COMMAND ----------

# MAGIC %md
# MAGIC **Logistic Regression**
# MAGIC BinaryClassificationEvaluator options:
# MAGIC - **areaUnderROC**
# MAGIC - **areaUnderPR**
# MAGIC 
# MAGIC 
# MAGIC The code for Logistic Regression look like this
# MAGIC 
# MAGIC `from pyspark.ml.evaluation import BinaryClassificationEvaluator`
# MAGIC 
# MAGIC `lrEvaluator = BinaryClassificationEvaluator().setPredictionCol("blrPrediction").setMetricName("areaUnderROC")`
# MAGIC 
# MAGIC The great advantage of Spark is that you are not limited on that prepared evaluation metrics. You can create your own metric, for example Lift on the first decile, and choose the best model on the basis of that custom made metric.

# COMMAND ----------

# MAGIC %md
# MAGIC ### IV. CrossValidator
# MAGIC 
# MAGIC As we mention above, CrossValidor needs three following items:
# MAGIC 
# MAGIC - Estimator
# MAGIC - Set of ParamMaps
# MAGIC - Evaluator
# MAGIC 
# MAGIC There are two more parameters that could have a significant impact mainly on the computing costs:
# MAGIC - **numFolds** - how many splits into training and testing datasets will be done, i.e. how many models will be trained for each parameters combination

# COMMAND ----------

# MAGIC %md
# MAGIC - **parallelism** - how many models will be trained in parallel

# COMMAND ----------

# MAGIC %md
# MAGIC So **how many models** will be calculated during cross validation? If we have two hyperametrs, each has three values and numFold is set to 3, in total we have
# MAGIC 2 x 3 x 3 = 18. In total 18 models will be trained during cross validation. ism parameter

# COMMAND ----------

# MAGIC %md
# MAGIC So finally we **set CrossValidator** ...

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator

lrCrossValidator = CrossValidator(estimator=lrPipeline,
                        estimatorParamMaps=lrParamGrid,
                        evaluator=lrEvaluator,
                        numFolds=2,
                        parallelism=3)

# COMMAND ----------

# MAGIC %md
# MAGIC ... and **train CrossValidator**.

# COMMAND ----------

lrCrossValidatorModel = lrCrossValidator.fit(train_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ____

# COMMAND ----------

# MAGIC %md
# MAGIC ## Result of CrossValidator

# COMMAND ----------

# MAGIC %md
# MAGIC Not user friendly CrossValidator output of performance

# COMMAND ----------

lrCrossValidatorModel.avgMetrics[0:10]

# COMMAND ----------

# MAGIC %md
# MAGIC Not user friendly parametr grid

# COMMAND ----------

lrParamGrid[0:5]

# COMMAND ----------

#getting param. names

l1 = len(lrParamGrid[0].keys()) 
paramName = []

for i in range(0,l1):
  paramName.append(list(lrParamGrid[0].keys())[i].name)

print(paramName)

# COMMAND ----------

#getting param. values

l2 = len(lrParamGrid)
paramValues = []

for i in range(0,l2):
  paramValues.append(lrParamGrid[i].values())

print(paramValues)

# COMMAND ----------

# all together

allLrModels = pd.DataFrame.from_records(paramValues, columns =paramName )
metricName = lrEvaluator.getMetricName()
allLrModels.loc[:,metricName] = lrCrossValidatorModel.avgMetrics

display(allLrModels.sort_values(allLrModels.columns[-1], ascending = True)[0:100])

# COMMAND ----------

# MAGIC %md
# MAGIC ### The best model

# COMMAND ----------

bestLr = LinearRegression(featuresCol="features", labelCol="label", predictionCol="lrPrediction", 
                                    regParam = 1.0, elasticNetParam = 0.1, maxIter = 5)

bestLrModel = bestLr.fit(train_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC By method **bestModel** we can find the best model calculating by crossValidator according to chosen metric (here RMSE)
# MAGIC 
# MAGIC `bestLrModel = lrCrossValidatorModel.bestModel`

# COMMAND ----------

lrPrediction = bestLrModel.transform(test_data_df)
print("Predictions:")
display(lrPrediction["lrPrediction", "label"].toPandas()[0:5])

# COMMAND ----------

intercept = bestLrModel.intercept
print(f"Intercept:\n{intercept}")

# COMMAND ----------

coef = bestLrModel.coefficients.toArray
print(f"Coefficients:\n{coef}")

# COMMAND ----------

#lets put the coefficients into the table
from itertools import chain

attrs = sorted((attr["idx"], attr["name"]) for attr in 
               (chain(*lrPrediction.schema["features"].metadata["ml_attr"]["attrs"].values())))

pairs = [(name, bestLrModel.coefficients[idx]) for idx, name in attrs]

sorted_pairs = sorted(pairs, key = lambda p: abs(p[1]), reverse=True)
variables = [("intercept", intercept)] + sorted_pairs

results = pd.DataFrame(variables, columns = ["predictors", "coefficients"])
display(results)

# COMMAND ----------


