#forked from : https://www.kaggle.com/fatmakursun/pyspark-ml-tutorial-for-beginners
from pyspark.sql.types import StructField,StructType,FloatType
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col
class MLProcessor:
    def __init__(self):
        pass
    def load_data(self) -> DataFrame:
        HOUSING_DATA = '../input/cal_housing.data'
        schema = StructType([
            StructField("long", FloatType(), nullable=True),
            StructField("lat", FloatType(), nullable=True),
            StructField("medage", FloatType(), nullable=True),
            StructField("totrooms", FloatType(), nullable=True),
            StructField("totbdrms", FloatType(), nullable=True),
            StructField("pop", FloatType(), nullable=True),
            StructField("houshlds", FloatType(), nullable=True),
            StructField("medinc", FloatType(), nullable=True),
            StructField("medhv", FloatType(), nullable=True)]
        )
        housing_df = spark.read.csv(path=HOUSING_DATA, schema=schema).cache()
        return housing_df
    def exploratory_analysis(self, housing_df:DataFrame):
        housing_df.take(5)
        housing_df.show(5)
        print(housing_df.columns)
        housing_df.printSchema()
        housing_df.select('pop', 'totbdrms').show(10)
        # group by housingmedianage and see the distribution
        result_df = housing_df.groupBy("medage").count().sort("medage", ascending=False)
        result_df.show(10)
        result_df.toPandas().plot.bar(x='medage', figsize=(14, 6))
        # Adjust the values of `medianHouseValue`
    def feature_preprocessing(self, housing_df:DataFrame):
        housing_df = housing_df.withColumn("medhv", col("medhv") / 100000)
        # Show the first 2 lines of `df`
        housing_df.show(2)
        # Add the new columns to `df`
        housing_df = (housing_df.withColumn("rmsperhh", F.round(col("totrooms") / col("houshlds"), 2))
                      .withColumn("popperhh", F.round(col("pop") / col("houshlds"), 2))
                      .withColumn("bdrmsperrm", F.round(col("totbdrms") / col("totrooms"), 2)))
        # Inspect the result
        housing_df.show(5)
        # Re-order and select columns
        housing_df = housing_df.select("medhv",
                                       "totbdrms",
                                       "pop",
                                       "houshlds",
                                       "medinc",
                                       "rmsperhh",
                                       "popperhh",
                                       "bdrmsperrm")
        featureCols = ["totbdrms", "pop", "houshlds", "medinc", "rmsperhh", "popperhh", "bdrmsperrm"]
        # put features into a feature vector column
        assembler = VectorAssembler(inputCols=featureCols, outputCol="features")
        assembled_df = assembler.transform(housing_df)
        assembled_df.show(10, truncate=False)
        # Initialize the `standardScaler`
        standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
        # Fit the DataFrame to the scaler
        scaled_df = standardScaler.fit(assembled_df).transform(assembled_df)
        # Inspect the result
        scaled_df.select("features", "features_scaled").show(10, truncate=False)
        # Split the data into train and test sets
        train_data, test_data = scaled_df.randomSplit([.8, .2], seed=rnd_seed)
        # Initialize `lr`
        lr = (LinearRegression(featuresCol='features_scaled', labelCol="medhv", predictionCol='predmedhv',
                               maxIter=10, regParam=0.3, elasticNetParam=0.8, standardization=False))
        # Fit the data to the model
        linearModel = lr.fit(train_data)
        # Coefficients for the model
        linearModel.coefficients
        linearModel.intercept
        coeff_df = pd.DataFrame({"Feature": ["Intercept"] + featureCols,
                                 "Co-efficients": np.insert(linearModel.coefficients.toArray(), 0,
                                                            linearModel.intercept)})
        coeff_df = coeff_df[["Feature", "Co-efficients"]]
        # Generate predictions
        predictions = linearModel.transform(test_data)
        # Extract the predictions and the "known" correct labels
        predandlabels = predictions.select("predmedhv", "medhv")
        predandlabels.show()
        # Get the RMSE
        '''
        The RMSE measures how much error there is between two datasets comparing a predicted value and an observed or known value.
        The smaller an RMSE value, the closer predicted and observed values are.
        '''
        print("RMSE: {0}".format(linearModel.summary.rootMeanSquaredError))
        '''
            The R2 ("R squared") or the coefficient of determination is a measure that shows how close the data are to the fitted regression line. This score will always be between 0 and a 100% (or 0 to 1 in this case), where 0% indicates that the model explains none of the variability of the response data around its mean, and 100% indicates the opposite: it explains all the variability.
            That means that, in general, the higher the R-squared, the better the model fits our data.
        '''
        print("MAE: {0}".format(linearModel.summary.meanAbsoluteError))
        # Get the R2
        print("R2: {0}".format(linearModel.summary.r2))

    def eval_model(self):
        evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='rmse')
        print("RMSE: {0}".format(evaluator.evaluate(predandlabels)))
        evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='mae')
        print("MAE: {0}".format(evaluator.evaluate(predandlabels)))
        evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='r2')
        print("R2: {0}".format(evaluator.evaluate(predandlabels)))

    def start_workflow(self):
