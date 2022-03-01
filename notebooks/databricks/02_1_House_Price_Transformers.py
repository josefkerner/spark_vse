# Databricks notebook source
# MAGIC %md
# MAGIC # <font color='red'>2. Spark and transformers</font>
# MAGIC Transfomers are the core of Spark "thinking" process. For better understanding it is best to start at begining: <b>How does Spark actually compute things in the code</b>?

# COMMAND ----------

# MAGIC %md
# MAGIC load the data

# COMMAND ----------

input_data = spark.read.parquet("/FileStore/cleanData")
input_data.createOrReplaceTempView("input_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## <font color='red'>2.1 Transformers</font>
# MAGIC So far we only did simple transformation query in SQL. But Spark can do more, much more actually.
# MAGIC 
# MAGIC **Documentation:** https://spark.apache.org/docs/2.3.0/ml-features.html
# MAGIC 
# MAGIC This is where <b><font color='red'>Transformes</font></b> come in. Transfomer is an object with two functions you can call:
# MAGIC - **Fit** (optional) - here are the data, look at them, find your parameters but **dont give me anything**
# MAGIC - **Transform** (always present) - here are the data, apply your transformation and **give me the result**
# MAGIC 
# MAGIC Let's further explain this using the first real transformer - **Imputer**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.1 Imputer
# MAGIC Imputer is transformer for filling out missing numerical values using mean or median.
# MAGIC 
# MAGIC **Example**: Let's say we have house Id without PriceNtile defined. Our decision (in reality not optimal) is that we want to fill these rows with median.

# COMMAND ----------

from pyspark.ml.feature import Imputer
imputer = Imputer(inputCols=['PriceNtile']
                  , outputCols=['PriceNtile_filled']
                  , strategy ="median")

# COMMAND ----------

# MAGIC %md
# MAGIC If we now call **.fit(dataset)** on this imputer, Spark finds what is the median of YearBuilt, but does not change data or return anything.

# COMMAND ----------

imputer_fitted = imputer.fit(input_data)

print(imputer_fitted)

# COMMAND ----------

# MAGIC %md
# MAGIC If we want to actually get the dataframe with a new column with the filled values (Spark in most cases creates new column rather than changing existing) we call **.transform(dataset)**. This returns the dataframe with a new column.

# COMMAND ----------

data_imputed = imputer_fitted.transform(input_data)

data_imputed.select("Id", "PriceNtile", "PriceNtile_filled").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.2 SQLTransformer
# MAGIC SQLTransformer lets you perform any SQL code on dataset send to it. It has only **.transform(dataset)** method as there are no parameters it needs 'find' for itself.
# MAGIC 
# MAGIC **Example:** Let's generalize outlier removal from first section to any dataset send to it.

# COMMAND ----------

from pyspark.ml.feature import SQLTransformer

sqlTransformer = SQLTransformer()\
  .setStatement("""
      SELECT *,
        case 
            when LotArea < 1450.0 then 1450.0 
            when LotArea > 17690.0 then 17690.0 
            else LotArea 
          end LotArea_bound
        , case 
            when YearBuilt < 1885.0 then 1885.0 
            when YearBuilt > 2069.0 then 2069.0 
            else YearBuilt 
          end YearBuilt_bound
        , Id
        , "ahoj " as const
      FROM __THIS__
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ** '__THIS__' ** is table you always select from, it will be the dataset you send to **.transform(dataset)** method.
# MAGIC 
# MAGIC Let's see the results:

# COMMAND ----------

data_transformed = sqlTransformer.transform(input_data)

data_transformed.select("Id", "LotArea","LotArea_bound", "YearBuilt", "YearBuilt_bound", "const").show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.3 QuantileDiscretizer
# MAGIC Splits a continious numerical variable into n buckets, based on quantiles.
# MAGIC 
# MAGIC **Example:** Let's split SalePrice column into 6 buckets.

# COMMAND ----------

from pyspark.ml.feature import QuantileDiscretizer

quantile_discretizer = QuantileDiscretizer(inputCol='SalePrice'
                  , outputCol='SalePrice_bucket'
                  , numBuckets = 6)

# COMMAND ----------

# MAGIC %md
# MAGIC Lets fit, transform and seee the output:

# COMMAND ----------

quantile_discretizer\
.fit(input_data)\
.transform(input_data)\
.select("Id", "SalePrice", "SalePrice_bucket")\
.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.4 StringIndexer
# MAGIC 
# MAGIC A label indexer that maps a string column of labels to an ML column of label indices. If the input column is numeric, we cast it to string and index the string values. The indices are in [0], numLabels), ordered by label frequencies. So the most frequent label gets index 0.
# MAGIC The simplest way to index is via the StringIndexer, which maps strings to different numerical IDs. Spark’s StringIndexer also creates metadata attached to the DataFrame that specify what inputs correspond to what outputs. This allows us later to get inputs back from their respective index values:
# MAGIC 
# MAGIC handleInvalid:
# MAGIC  - skip (which will filter out rows with bad values)
# MAGIC  - error (which will throw an error)
# MAGIC  - keep (creates new index)
# MAGIC  
# MAGIC **Example:** Let's index categorical column MSSubClass.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol = "MSSubClass"
                            ,outputCol='MSSubClass_index'
                            ,handleInvalid="keep")

# COMMAND ----------

# MAGIC %md
# MAGIC And see the result:

# COMMAND ----------

data_indexed = indexer.fit(input_data).transform(input_data)

data_indexed \
    .select("Id", "MSSubClass", "MSSubClass_index") \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.5 OneHotEncoderEstimator
# MAGIC 
# MAGIC   - Indexing categorical variables is only half of the story. One-hot encoding is an extremely common data transformation performed after indexing categorical variables. This is because indexing does not always represent our categorical variables in the correct way for downstream models to process. For instance, when we index our “color” column, you will notice that some colors have a higher value (or index number) than others (in our case, blue is 1 and green is 2).
# MAGIC   - This is incorrect because it gives the mathematical appearance that the input to the machine learning algorithm seems to specify that green > blue, which makes no sense in the case of the current categories. To avoid this, we use OneHotEncoder, which will convert each distinct value to a Boolean flag (1 or 0) as a component in a vector. When we encode the color value, then we can see these are no longer ordered, making them easier for downstream models (e.g., a linear model) to process
# MAGIC   - Input column must be of type NumericType
# MAGIC   
# MAGIC **Example:** One hot encode our indexed variable MSSubClass_indexed.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCols = ["MSSubClass_index"],
                                 outputCols=["MSSubClass_index_vec"])

# COMMAND ----------

# MAGIC %md
# MAGIC And see the result:

# COMMAND ----------

encoder \
    .fit(data_indexed) \
    .transform(data_indexed)\
    .select("Id", "MSSubClass", "MSSubClass_index", "MSSubClass_index_vec") \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.6 VectorAssembler
# MAGIC  - One of the most important transforms in Spark. Most models do not take as an input a list of columns and what to predict, but they need one vector with information from all columns.
# MAGIC  - VectorAssembler takes as an input the list of columns a returns a new column with a vector.
# MAGIC   
# MAGIC **Example:** Join three numerical columns LotArea, YearBuilt, YearRemodAdd into vector.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["LotArea", "YearBuilt", "YearRemodAdd"], 
                            outputCol="Num_vector")

# COMMAND ----------

# MAGIC %md
# MAGIC See the output:

# COMMAND ----------

assembler \
    .transform(input_data)\
    .select("Id", "LotArea", "YearBuilt", "YearRemodAdd", "Num_vector") \
    .show(20, False)

# COMMAND ----------


