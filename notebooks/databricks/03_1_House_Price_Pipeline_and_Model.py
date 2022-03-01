# Databricks notebook source
# MAGIC %md
# MAGIC In the transformer section we always used **.fit(dataset).transform(dataset)** for every transformer. This seems quite redundant, why not put all transformers together and just call it once? That is the role of pipeline.
# MAGIC 
# MAGIC #### Pipeline is collection of transformers
# MAGIC 
# MAGIC Apart from simplifying the code it adds greatly to reusability.
# MAGIC  - Imagine we are preparing data for model training and we train the model on January dataset.
# MAGIC  - Now February ends and we want to score the new dataset.
# MAGIC      - The new data are unprepared
# MAGIC      - Without pipeline we would need to again prepare them for the model
# MAGIC      - But with pipeline we just save pipeline during training and when it is time to score, we load it.
# MAGIC      - Whole data preparation becomes just two lines
# MAGIC 
# MAGIC ## 3.1 How can we create a pipeline?
# MAGIC Pipeline creation is simple. At begining we create a Python list with transformers in order and pipeline is then created with just one line.
# MAGIC 
# MAGIC **Example:** We take categorical value HouseStyle, we want to string-index it and on-hot-encode it.
# MAGIC 
# MAGIC First we create session and load the data:

# COMMAND ----------

input_data = spark.read.parquet("/FileStore/cleanData/")

# COMMAND ----------

int_c = [c for c in input_data.columns]

# COMMAND ----------

# MAGIC %md
# MAGIC Now lets see the normal way of doing this without a pipeline:

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol = "HouseStyle"
                            ,outputCol='HouseStyle_index'
                            ,handleInvalid="keep")

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCols = ["HouseStyle_index"],
                                 outputCols=["HouseStyle_index_vec"])

data_indexed = indexer.fit(input_data).transform(input_data)
data_encoded = encoder.fit(data_indexed).transform(data_indexed)

data_encoded.select("Id", "HouseStyle", "HouseStyle_index","HouseStyle_index_vec").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's do it with a pipeline that will look like this:

# COMMAND ----------

from pyspark.ml import Pipeline

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol = "HouseStyle"
                            ,outputCol='HouseStyle_index'
                            ,handleInvalid="keep")

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCols = ["HouseStyle_index"],
                                 outputCols=["HouseStyle_index_vec"])

my_pipeline = Pipeline(stages = [indexer, encoder])
data_transformed = my_pipeline.fit(input_data).transform(input_data)

data_transformed.select("Id", "HouseStyle", "HouseStyle_index","HouseStyle_index_vec").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Creating whole pipeline for our data
# MAGIC Now we know what pipeline is. We can combine it with transformers from previous section and create whole pipeline for our dataset.

# COMMAND ----------

pipeline_stages = []

numericalFeatures = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'OpenPorchSF']
categoricalFeatures = ['MSSubClass', 'MSZoning', 'HouseStyle', 'OverallQual', 'OverallCond', 'ExterQual', 'Foundation', 
                       'BsmtQual', 'BsmtCond', 'HeatingQC', 'CentralAir', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 
                       'GarageCars', 'GarageQual', 'PoolQC', 'Neighborhood']

from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer, OneHotEncoder, StringIndexer, VectorAssembler

# Remove outliers
sqlTransformer = SQLTransformer()\
  .setStatement("""
      SELECT
        *
        , case 
            when LotArea < 1450.0 then 1450.0 
            when LotArea > 17690.0 then 17690.0 
            else LotArea 
          end LotArea_bound
        , case 
            when YearBuilt < 1885.0 then 1885.0 
            when YearBuilt > 2069.0 then 2069.0 
            else YearBuilt 
          end YearBuilt_bound
        , case 
            when YearRemodAdd < 1911.5 then 1911.5 
            when YearRemodAdd > 2059.5 then 2059.5 
            else YearRemodAdd 
          end YearRemodAdd_bound
        , case 
            when TotalBsmtSF < 40.5 then 40.5
            when TotalBsmtSF > 2052.5 then 2052.5 
            else TotalBsmtSF 
          end TotalBsmtSF_bound
        , case 
            when GrLivArea < 156.0 then 156.0 
            when GrLivArea > 2748.0 then 2748.0 
            else GrLivArea 
          end GrLivArea_bound
        , case 
            when GarageArea < -39.0 then -39.0 
            when GarageArea > 945.0 then 945.0 
            else GarageArea 
          end GarageArea_bound
        , case 
            when OpenPorchSF < -102.0 then -102.0 
            when OpenPorchSF > 170.0 then 170.0 
            else OpenPorchSF 
          end OpenPorchSF_bound
        , cast(
              case 
                when SalePrice < 3750.0 then 3750.0
                when SalePrice > 340150.0 then 340150.0 
                else SalePrice 
               end as double) SalePrice_bound
        , SalePrice label
      FROM __THIS__
    """)

pipeline_stages.append(sqlTransformer)

# Index all categorical features
for c in categoricalFeatures:
    indexer = StringIndexer(inputCol = c ,outputCol = c + "_index", handleInvalid = "keep")
    pipeline_stages.append(indexer)

# One-hot encode all indexed variables
encoder = OneHotEncoder(inputCols = [(x + "_index") for x in categoricalFeatures],
                                 outputCols=[(x + "_index_vec") for x in categoricalFeatures])

pipeline_stages.append(encoder)

# Assemble one-hot encoded variables with numerical features with removed outliers
assembler = VectorAssembler(inputCols=[(x + "_index_vec") for x in categoricalFeatures] 
                                + [(x + "_bound") for x in numericalFeatures], 
                            outputCol="features")

pipeline_stages.append(assembler)


# COMMAND ----------

pipeline_stages

# COMMAND ----------

# Create pipeline and fit it
pipeline = Pipeline(stages = pipeline_stages)
final_data = pipeline \
                .fit(input_data) \
                .transform(input_data) \
                .select("Id", "label","features")

final_data.write.mode("overwrite").parquet("/FileStore/final_data")

final_data.show(1, False)

# COMMAND ----------

input_data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## <font color='orange'>3.3 Linear Regression</font>
# MAGIC We have prepared data for the model training. We can now use them to train a simple Linear regression.
# MAGIC 
# MAGIC Basic info about Linear Regression:
# MAGIC * Predicts continuous variable
# MAGIC * Easy to interpret the results
# MAGIC * Sensitive to outliers
# MAGIC * **Documentation:** https://spark.apache.org/docs/2.3.0/ml-classification-regression.html#linear-regression 
# MAGIC 
# MAGIC Predicted variable is a linear equation:
# MAGIC ## $$ y_i =\beta_0+\beta_1x_1 ... $$

# COMMAND ----------

# MAGIC %md
# MAGIC First we define the model, lets use default parameters.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features', labelCol='label')

# COMMAND ----------

# MAGIC %md
# MAGIC Models are **transformers** too, so to train we call **.fit(dataset)**

# COMMAND ----------

lr_model = lr.fit(final_data)

# COMMAND ----------

# MAGIC %md
# MAGIC To score dateset we call **.transform**. Normally we would split dataset on test/train, right now we just use one dataset for training and scoring:

# COMMAND ----------

lr_scored = lr_model.transform(final_data)

# COMMAND ----------

lr_scored.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3.1 Model evaluation
# MAGIC With trained model we can evalute it's performance. Performance metric are different for regression/classification and even different for individual algorithms. Here we try to use the most common.
# MAGIC 
# MAGIC **Coefficients of equation + intercept**

# COMMAND ----------

lr_model.coefficients

# COMMAND ----------

lr_model.intercept

# COMMAND ----------

from itertools import chain
import pandas as pd

attrs = sorted((attr["idx"], attr["name"]) for attr in 
               (chain(*lr_scored.schema["features"].metadata["ml_attr"]["attrs"].values())
               ))

pairs = [(name, lr_model.coefficients[idx]) for idx, name in attrs]
sorted_pairs = sorted(pairs, key = lambda p: abs(p[1]), reverse=True)
variables = sorted_pairs

dataset = pd.DataFrame(variables, columns = ["predictors", "coefficients"]) 

display(dataset)

# COMMAND ----------

lr_model.intercept

# COMMAND ----------

# MAGIC %md
# MAGIC **RMSE** - RootMeanSquaredError
# MAGIC - The RMSE measures how much error there is between two datasets comparing a predicted value and an observed or known value.
# MAGIC         The smaller an RMSE value, the closer predicted and observed values are.

# COMMAND ----------

lr_model.summary.rootMeanSquaredError

# COMMAND ----------

# MAGIC %md
# MAGIC **r^2** - R squared
# MAGIC - The R2 ("R squared") or the coefficient of determination is a measure that shows how close the data are to the fitted regression line. This score will always be between 0 and a 100% (or 0 to 1 in this case), where 0% indicates that the model explains none of the variability of the response data around its mean, and 100% indicates the opposite: it explains all the variability.
# MAGIC             That means that, in general, the higher the R-squared, the better the model fits our data.

# COMMAND ----------

lr_model.summary.r2
