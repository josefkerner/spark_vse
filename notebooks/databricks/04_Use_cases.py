# Databricks notebook source
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("use_cases").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. PTB model in telco
# MAGIC For this project we developed near complete automated pipeline. Here is one example of custom transformer, it's called Sampling transformer and is used to oversampling/undersampling data for classification.

# COMMAND ----------

from pyspark.ml import Transformer, Estimator, Pipeline
from pyspark.sql.functions import column as col
from pyspark.sql.functions import explode, array, lit

class SamplingTransformer(Transformer):
    
    def __init__(self, labelCol = 'label', split = [0.7, 0.3], oversampling = 10, undersampling = 1, minority = 1):
        """ This is a custom oversampling transformer used to oversample positive or negative observation 
        
        Attributes:
            labelCol (str): Columns with target variable. Defaults to label
            split (list(float)): Split to be used for the train/test ratio. Defaults to [0.7, 0.3].
            oversampling (int): Factor of the oversampling. Defaults to 10.
            undersampling (int): Factor of the undersampling. Defaults to 1.
            minority (int): Which of the observation is the minority to be sampled. Defaults to 1.
        """ 
        
        super(SamplingTransformer, self).__init__()
        
        # Assign the constructor varibles
        self.labelCol = labelCol
        self.split = split
        self.oversampling = oversampling
        
        # Check if undersampling is smaller than one
        if undersampling < 1:
          print("undersampling must be greater than 1")
          undersampling = 1
          
        self.undersampling = undersampling
        self.minority = minority

    def sample(self, dataset):
        """ This function takes input dataset and does the sampling based upon the inpur parameters."""
        
        # Create list in range of the sample factor
        samples = range(0, self.oversampling)
        
        # Create a new column 'dummy' with array in range of samples and explode it. This is the fastest way to oversample in pySpark.
        if self.minority == 1:
            
          # Split the positive observations
          train_data_positive,test_data_positive = dataset.filter(col(self.labelCol) == 1).randomSplit([0.7,0.3])
        
          # Split the negative observations
          train_data_negative,test_data_negative = dataset.filter(col(self.labelCol) == 0).randomSplit([(0.7/self.undersampling), 0.3])
          
          train_data_positive = train_data_positive.withColumn("dummy", explode(array([lit(x) for x in samples]))).drop("dummy")
        
        else:
          
          # Split the positive observations
          train_data_positive,test_data_positive = dataset.filter(col(self.labelCol) == 1).randomSplit([(0.7/self.undersampling),0.3])
        
          # Split the negative observations
          train_data_negative,test_data_negative = dataset.filter(col(self.labelCol) == 0).randomSplit([0.7, 0.3])
          
          train_data_negative = train_data_negative.withColumn("dummy", explode(array([lit(x) for x in samples]))).drop("dummy")
        
        # Union positive and negative train/test dataframes
        train_data = train_data_negative.union(train_data_positive)
        test_data = test_data_negative.union(test_data_positive)
        
        return (train_data, test_data)
        
    def _transform(self, dataset):
        
        return self.sample(dataset)

# COMMAND ----------

raw_data = spark.read.parquet('/appl/wsp_data_science_workshop/cleanData_churn')

# COMMAND ----------

# MAGIC %md
# MAGIC See how much of each target variable we have:

# COMMAND ----------

raw_data.groupBy('label').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Now we want to oversample the label = 1 by factor of 2:

# COMMAND ----------

st = SamplingTransformer(oversampling=3)

train_data, test_data = st.transform(raw_data)

# COMMAND ----------

train_data.groupBy('label').count().show()

# COMMAND ----------

test_data.groupBy('label').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Recommendation engine - usage of config
# MAGIC Engine need to run for different segments. Rather than duplicating and making small changes to code, we use Config file.

# COMMAND ----------

from configparser import ConfigParser, ExtendedInterpolation

config = ConfigParser(interpolation=ExtendedInterpolation())

config.read('config/config.ini')

# COMMAND ----------

config['MicroClustering']['output_table']

# COMMAND ----------

config['MasterParameters']['date_valid']

# COMMAND ----------


