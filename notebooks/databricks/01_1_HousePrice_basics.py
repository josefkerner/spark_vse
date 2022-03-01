# Databricks notebook source
# MAGIC %md
# MAGIC # House prices analysis
# MAGIC 
# MAGIC This is the first excercise of DS workshop.
# MAGIC 
# MAGIC ## About JupyterHub
# MAGIC As you can see, we are using *JupyterHub* environment: To learn more you can visit the official website: https://jupyter.org/hub or a cheat sheet: https://medium.com/edureka/jupyter-notebook-cheat-sheet-88f60d1aca7.
# MAGIC 
# MAGIC ## About MD (markdown)
# MAGIC We are using MD to describe our notebooks, more information at: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
# MAGIC 
# MAGIC ## About Python
# MAGIC During this course we will be using Python as programming language for all data science related tasks. Python basics: https://datacamp-community-prod.s3.amazonaws.com/e30fbcd9-f595-4a9f-803d-05ca5bf84612
# MAGIC 
# MAGIC ## About Pandas
# MAGIC Pandas is Python framework we will be using for data manipulation and ransformation. Cheat sheet at: https://datacamp-community-prod.s3.amazonaws.com/e30fbcd9-f595-4a9f-803d-05ca5bf84612
# MAGIC 
# MAGIC ## About Spark
# MAGIC Apache Spark is highly scalable framework we will be using for the data science related tasks. More information at: https://github.com/runawayhorse001/CheatSheet/raw/master/cheatSheet_pyspark.pdf, https://spark.apache.org/docs/latest/ml-guide.html
# MAGIC 
# MAGIC <hr />

# COMMAND ----------

# MAGIC %md
# MAGIC #### Download the data 
# MAGIC - Link : https://www.kaggle.com/lespin/house-prices-dataset
# MAGIC - extract it and place file train.csv into location /FileStore

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing packages
# MAGIC At first we must import all the packages we will be using.

# COMMAND ----------

# MAGIC %md
# MAGIC Spark libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.mllib.stat import Statistics
from pyspark.sql.types import *
from pyspark.sql.functions import col, countDistinct, count, when, isnan, expr

# COMMAND ----------

# MAGIC %md
# MAGIC Python libraries

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml import Pipeline

#temporary hide error reports
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading CSV data
# MAGIC 
# MAGIC ### Spark and Pandas
# MAGIC Throughout the course we will be using 2 frameworks for the data manipulation:
# MAGIC   - Spark: `spark.read.csv("somedata")`
# MAGIC   - Pandas: `pd.read_csv("somedata")`
# MAGIC   
# MAGIC Both frameworks use so called `dataframes`.<br />
# MAGIC ![dataframe](https://media.geeksforgeeks.org/wp-content/uploads/finallpandas.png)
# MAGIC 
# MAGIC More informationm about reading CSV with Spark: https://docs.databricks.com/data/data-sources/read-csv.html
# MAGIC 
# MAGIC `spark.read.csv("somedata")`
# MAGIC 
# MAGIC   - we can tell spark to use header via: `.option("header", "true")`
# MAGIC   - or to assess schema for us via: `.option("inferSchema", "true")`
# MAGIC   
# MAGIC Many times we will exchange data between these 2 frameworks. Why?
# MAGIC   - Pandas: is great for **small data** manipulation. Such as: data visualization, writing data to screen, simple transformations
# MAGIC   - Spark: is great for **big data** manipulation. Such as: running analytical models, complex transformations
# MAGIC 
# MAGIC ### Transform dataframes between Spark and Pandas   
# MAGIC To transform spark dataframe to Pandas dataframe use: `sparkDf.toPandas()`
# MAGIC 
# MAGIC To transform Pandas dataframe to spark dataframe use: `spark.createDataFrame(pdDataframe)`
# MAGIC 
# MAGIC <hr />
# MAGIC   
# MAGIC We will only use a subset of attributes, so we are using `.select()` - other attributes may be interresting for further analysis, but for now we are fine with 25 attributes

# COMMAND ----------

trainDataPath = "/FileStore/train.csv"
rawData = spark \
    .read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(trainDataPath)\
    .select( 'Id', 'SalePrice', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 
            'GrLivArea', 'GarageArea', 'OpenPorchSF', 'MSSubClass', 'MSZoning', 
            'Neighborhood', 'HouseStyle', 'OverallQual', 'OverallCond',
            'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'HeatingQC',
            'CentralAir', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces',
            'GarageCars', 'GarageQual', 'PoolQC')

# COMMAND ----------

# MAGIC %md
# MAGIC We can use Pandas framework to display our DataFrame.
# MAGIC 
# MAGIC Also we could you spark\`s `.show()` method, but Pandas looks better in Jupyetr notebooks
# MAGIC 
# MAGIC Our data looks like this:

# COMMAND ----------

rawData.limit(10).toPandas()

# COMMAND ----------

rawData.show(5, False)

# COMMAND ----------

rawData.createOrReplaceTempView("data")
spark.sql('''select * from data''').limit(10).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC **LotArea**:
# MAGIC     _Lot size in square feet_
# MAGIC     
# MAGIC **YearBuilt**:
# MAGIC     _Original construction date_
# MAGIC     
# MAGIC **YearRemodAdd**:
# MAGIC     _Remodel date (same as construction date if no remodeling or additions)_
# MAGIC     
# MAGIC **TotalBsmtSF**:
# MAGIC     _Total square feet of basement area_
# MAGIC     
# MAGIC **GrLivArea**:
# MAGIC     _Above grade (ground) living area square feet_
# MAGIC 
# MAGIC **GarageArea**:
# MAGIC     _Size of garage in square feet_
# MAGIC     
# MAGIC **OpenPorchSF**:
# MAGIC     _Open porch area in square feet_
# MAGIC     
# MAGIC **MSSubClass**:
# MAGIC     _Identifies the type of dwelling involved in the sale._
# MAGIC   - **20**: _1-STORY 1946 & NEWER ALL STYLES_
# MAGIC   - **30**: _1-STORY 1945 & OLDER_
# MAGIC   - **40**: _1-STORY W/FINISHED ATTIC ALL AGES_
# MAGIC   - **45**: _1-1/2 STORY - UNFINISHED ALL AGES_
# MAGIC   - **50**: _1-1/2 STORY FINISHED ALL AGES_
# MAGIC   - **60**: _2-STORY 1946 & NEWER_
# MAGIC   - **70**: _2-STORY 1945 & OLDER_
# MAGIC   - **75**: _2-1/2 STORY ALL AGES_
# MAGIC   - **80**: _SPLIT OR MULTI-LEVEL_
# MAGIC   - **85**: _SPLIT FOYER_
# MAGIC   - **90**: _DUPLEX - ALL STYLES AND AGES_
# MAGIC   - **120**: _1-STORY PUD (Planned Unit Development) - 1946 & NEWER_
# MAGIC   - **150**: _1-1/2 STORY PUD - ALL AGES_
# MAGIC   - **160**: _2-STORY PUD - 1946 & NEWER_
# MAGIC   - **180**: _PUD - MULTILEVEL - INCL SPLIT LEV/FOYER_
# MAGIC   - **190**: _2 FAMILY CONVERSION - ALL STYLES AND AGES_
# MAGIC        
# MAGIC **OverallQual**:
# MAGIC     _Rates the overall material and finish of the house_
# MAGIC     
# MAGIC **OverallCond**:
# MAGIC     _Rates the overall condition of the house_
# MAGIC   - **10**: _Very Excellent_
# MAGIC   - **9**: 	_Excellent_
# MAGIC   - **8**: 	_Very Good_
# MAGIC   - **7**: 	_Good_
# MAGIC   - **6**: 	_Above Average_
# MAGIC   - **5**: 	_Average_
# MAGIC   - **4**: 	_Below Average_
# MAGIC   - **3**: 	_Fair_
# MAGIC   - **2**: 	_Poor_
# MAGIC   - **1**: 	_Very Poor_
# MAGIC   
# MAGIC **TotRmsAbvGrd**:
# MAGIC 	_Total rooms above grade (does not include bathrooms)_
# MAGIC 
# MAGIC **Fireplaces**:
# MAGIC 	_Number of fireplaces_
# MAGIC 
# MAGIC **GarageCars**:
# MAGIC 	_Size of garage in car capacity_
# MAGIC 
# MAGIC **Neighborhood**:
# MAGIC 	_sorted by avg price - 1-highest, 25-lowest_
# MAGIC 
# MAGIC **MSZoning**:
# MAGIC 	_Identifies the general zoning classification of the sale._
# MAGIC   - **A**: 	_Agriculture_
# MAGIC   - **C**: 	_Commercial_
# MAGIC   - **FV**: 	_Floating Village Residential_
# MAGIC   - **I**: 	_Industrial_
# MAGIC   - **RH**: 	_Residential High Density_
# MAGIC   - **RL**: 	_Residential Low Density_
# MAGIC   - **RP**: 	_Residential Low Density Park_
# MAGIC   - **RM**: 	_Residential Medium Density_
# MAGIC 	   
# MAGIC **HouseStyle**:
# MAGIC 	_Style of dwelling_
# MAGIC   - **1Story**: 	_One story_
# MAGIC   - **1.5Fin**: 	_One and one-half story: 2nd level finished_
# MAGIC   - **1.5Unf**: 	_One and one-half story: 2nd level unfinished_
# MAGIC   - **2Story**: 	_Two story_
# MAGIC   - **2.5Fin**: 	_Two and one-half story: 2nd level finished_
# MAGIC   - **2.5Unf**: 	_Two and one-half story: 2nd level unfinished_
# MAGIC   - **SFoyer**: 	_Split Foyer_
# MAGIC   - **SLvl**: 	_Split Level_
# MAGIC        
# MAGIC **ExterQual**:
# MAGIC 	_Evaluates the quality of the material on the exterior_
# MAGIC   - **Ex:** 	_Excellent_
# MAGIC   - **Gd:** 	_Good_
# MAGIC   - **TA:** 	_Average/Typical_
# MAGIC   - **Fa:** 	_Fair_
# MAGIC   - **Po:** 	_Poor_
# MAGIC        
# MAGIC **Foundation**:
# MAGIC 	_Type of foundation_
# MAGIC   - **BrkTil**: 	_Brick & Tile_
# MAGIC   - **Cblock**: 	_Cinder Block_
# MAGIC   - **Pconc**: 	_Poured Contrete_	
# MAGIC   - **Slab**: 	_Slab_
# MAGIC   - **Stone**: 	_Stone_
# MAGIC   - **Wood**: 	_Wood_
# MAGIC        
# MAGIC **BsmtQual**:
# MAGIC 	_Evaluates the height of the basement_
# MAGIC   - **Ex**	_Excellent (100+ inches)_
# MAGIC   - **Gd**	_Good (90-99 inches)_
# MAGIC   - **TA**	_Typical (80-89 inches)_
# MAGIC   - **Fa**	_Fair (70-79 inches)_
# MAGIC   - **Po**	_Poor (<70 inches_
# MAGIC   - **NA**	_No Basement_
# MAGIC 
# MAGIC **BsmtCond**:
# MAGIC 	_Evaluates the general condition of the basement_
# MAGIC   - **Ex:** _Excellent_
# MAGIC   - **Gd:** _Good_
# MAGIC   - **TA:** _Typical - slight dampness allowed_
# MAGIC   - **Fa:** _Fair - dampness or some cracking or settling_
# MAGIC   - **Po:** _Poor - Severe cracking, settling, or wetness_
# MAGIC   - **NA:** _No Basement_
# MAGIC        
# MAGIC **HeatingQC**:
# MAGIC 	_Heating quality and condition_
# MAGIC   - **Ex:** _Excellent_
# MAGIC   - **Gd:** _Good_
# MAGIC   - **TA:** _Average/Typical_
# MAGIC   - **Fa:** _Fair_
# MAGIC   - **Po:** _Poor_
# MAGIC        
# MAGIC **CentralAir**:
# MAGIC 	_Central air conditioning_
# MAGIC   - **N**: 	_No_
# MAGIC   - **Y**: 	_Yes_
# MAGIC 
# MAGIC **KitchenQual**:
# MAGIC 	_Kitchen quality_
# MAGIC   - **Ex**: _Excellent_
# MAGIC   - **Gd**: _Good_
# MAGIC   - **TA**: _Typical/Average_
# MAGIC   - **Fa**: _Fair_
# MAGIC   - **Po**: _Poor_
# MAGIC        
# MAGIC **GarageQual**:
# MAGIC 	_Garage quality_
# MAGIC   - **Ex**: _Excellent_
# MAGIC   - **Gd**: _Good_
# MAGIC   - **TA**: _Typical/Average_
# MAGIC   - **Fa**: _Fair_
# MAGIC   - **Po**: _Poor_
# MAGIC   - **NA**: _No Garage_
# MAGIC        
# MAGIC **PoolQC**:
# MAGIC 	_Pool quality_
# MAGIC   - **Ex**: _Excellent_
# MAGIC   - **Gd**: _Good_
# MAGIC   - **TA**: _Average/Typical_
# MAGIC   - **Fa**: _Fair_
# MAGIC   - **NA**: _No Pool_

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL reprocessing
# MAGIC Change Neighborhood name from original value to rank ordered by average price of houses in that Neighborhood<br>
# MAGIC Neighborhood names are hard to evaluate for us, but rank is much easier to work with
# MAGIC 
# MAGIC Also we add price range which is `NTILE(10)` ordered by SalePrice

# COMMAND ----------


cleanData = spark.sql("""
select
    a.*
    , cast(b.Neighborhood_new as string) as Neighborhood_new
    , cast(ntile(10) over (order by SalePrice) as double) PriceNtile
from
    data a
left join
    (
        select 
            Neighborhood, 
            rank() over (order by avg(SalePrice) desc) Neighborhood_new
        from 
            data
        group by
            1
    ) b using (Neighborhood)
""").drop("Neighborhood").withColumnRenamed("Neighborhood_new", "Neighborhood")

# COMMAND ----------

# MAGIC %md
# MAGIC You can do with spark dataframe all operation that you know from SQL:
# MAGIC 
# MAGIC **WHERE**

# COMMAND ----------

spark.sql('''select count(1) as cnt from data where SalePrice >= 150000''').show()

# COMMAND ----------

cleanData.where(col("SalePrice") >= 150000).count()

# COMMAND ----------

# MAGIC %md
# MAGIC **GROUP BY**

# COMMAND ----------

spark.sql('''select YearBuilt, count(1) as cnt from data group by YearBuilt order by YearBuilt''').show(5)

# COMMAND ----------

cleanData.groupBy("YearBuilt").count().orderBy("YearBuilt").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Analytical funcions**

# COMMAND ----------

spark.sql('''
select 
    YearBuilt
    , SalePrice
    , avg(SalePrice) over (partition by YearBuilt) as avg_price_per_year 
from data order by YearBuilt''').show(10)

# COMMAND ----------

cleanData.select("YearBuilt", "SalePrice", expr("avg(SalePrice) over (partition by YearBuilt)").alias("avg_price_per_year"))\
    .orderBy("YearBuilt").show(10) 

# COMMAND ----------

# MAGIC %md
# MAGIC **JOIN**

# COMMAND ----------

col1 = ["Id", "SalePrice", "LotArea", "YearBuilt"]
col2 = ["Id", "YearRemodAdd", "TotalBsmtSF", "GrLivArea"]

df1 = cleanData.select(col1)
df2 = cleanData.select(col2)

df1.join(df2, ["Id"]).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Something Spark can do even better
# MAGIC 
# MAGIC **Columns processing**
# MAGIC 
# MAGIC We want to sum all area columns

# COMMAND ----------

spark.sql('''
select LotArea + GrLivArea + GarageArea as area_number 
from data order by Id''').show(5)

# COMMAND ----------

area_cols = [c for c in cleanData.columns if "Area" in c]

cleanData.orderBy("Id").select(sum(col(c) for c in area_cols).alias("area_number")).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Another example**
# MAGIC 
# MAGIC We want to calculate the distinct number of values for each columns

# COMMAND ----------

spark.sql('''
select 
    count(distinct Id) as id 
    ,count(distinct SalePrice) as SalePrice 
    ,count(distinct LotArea) as LotArea 
    ,count(distinct YearBuilt) as YearBuilt
    ,count(distinct YearRemodAdd) as YearRemodAdd
    ,count(distinct TotalBsmtSF) as TotalBsmtSF
    ,count(distinct GrLivArea) as GrLivArea
    ,count(distinct GarageArea) as GarageArea
    ,count(distinct OpenPorchSF) as OpenPorchSF
    ,count(distinct MSSubClass) as MSSubClass
    ,count(distinct MSZoning) as MSZoning
    ,count(distinct HouseStyle) as HouseStyle
    ,count(distinct OverallQual) as OverallQual
    ,count(distinct OverallCond) as OverallCond
    ,count(distinct ExterQual) as ExterQual
    ,count(distinct Foundation) as Foundation
    ,count(distinct BsmtQual) as BsmtQual
    ,count(distinct BsmtCond) as BsmtCond
    ,count(distinct HeatingQC) as HeatingQC
    ,count(distinct CentralAir) as CentralAir
    ,count(distinct KitchenQual) as KitchenQual
    ,count(distinct TotRmsAbvGrd) as TotRmsAbvGrd
    ,count(distinct Fireplaces) as Fireplaces
    ,count(distinct GarageCars) as GarageCars
    ,count(distinct GarageQual) as GarageQual
    ,count(distinct PoolQC) as PoolQC
    ,count(distinct Neighborhood) as Neighborhood
from data''').show()

# COMMAND ----------

cleanData.agg(*(countDistinct(col(c)).alias(c) for c in cleanData.columns)).show()

# COMMAND ----------

featureDistinctCounts = cleanData.agg(*(countDistinct(col(c)).alias(c) for c in cleanData.columns)).toPandas().transpose()
featureDistinctCounts.columns=['distinctCount']
rowCount = cleanData.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Storing the data on disk
# MAGIC 
# MAGIC We write data as parquet

# COMMAND ----------

cleanData.write.mode('overwrite').parquet("/FileStore/cleanData")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable distribution
# MAGIC 
# MAGIC First of all we will have a look at data types assessed by spark.
# MAGIC We create 4 arrays: `integerFeatures`, `stringFeatures`, `doubleFeatures`, `otherTypeFeatures` and asign each feature to one of them.
# MAGIC 
# MAGIC next we will assign features to one of these groups:
# MAGIC 
# MAGIC   - **Categorical feature**: Categorical variables contain a finite number of categories or distinct groups. Categorical data might not have a logical order. For example, categorical predictors include gender, material type, and payment method.
# MAGIC 
# MAGIC   - **Continuous(numeric) feature**: Continuous variables are numeric variables that have an infinite number of values between any two values. A continuous variable can be numeric or date/time. For example, the length of a part or the date and time a payment is received.
# MAGIC   
# MAGIC ![feature types](https://cdn.shopify.com/s/files/1/1334/2321/articles/Picture1.png?v=1497575369)

# COMMAND ----------

# MAGIC %md
# MAGIC For further demonstration, we choose a few numerical and categorical variables.

# COMMAND ----------

numericalFeatures = ["LotArea", "YearBuilt", "GarageArea"]
categoricalFeatures =  ["MSZoning", "HouseStyle"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualizations 
# MAGIC We are using Seaborn/Matplotlib for this visualization. To learn more see: https://seaborn.pydata.org/
# MAGIC 
# MAGIC Why to use Python for visualisation:
# MAGIC - all setting in one place
# MAGIC - scalability
# MAGIC - repeatable usage 
# MAGIC 
# MAGIC **color palette**: we re using `Set2` palette, but you can pick any palette you like. To learn more: https://seaborn.pydata.org/tutorial/color_palettes.html
# MAGIC 
# MAGIC ## Histograms
# MAGIC We use histogram to visualize distribution of numeric features

# COMMAND ----------

clr = "#c2c1f2"
pltt = "Set2"
sns.set_palette(pltt)
plt.rcParams.update({'figure.max_open_warning': 0})

# COMMAND ----------

def histogram(x, bins):
  plt.figure()
  sns.set_style("darkgrid")
  f, ax = plt.subplots(figsize=(12, 6))
  sns.distplot(x, bins = bins, kde = True )
  plt.grid(axis = 'x')
  return;

for c in numericalFeatures: 
  histogram(cleanData.toPandas()[c].dropna(), 20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation matrix
# MAGIC 
# MAGIC Another great way to visualize relationship between variables is correlation matrix

# COMMAND ----------

corrmat = cleanData.toPandas().corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cmap="YlGnBu", annot=True);

# COMMAND ----------



# COMMAND ----------


