# Databricks notebook source
# MAGIC %md
# MAGIC # Partitioning

# COMMAND ----------

data = spark.sql("select transaction_id, first(post_date) as post from mir_dwkb.posted_transaction where post_date = '2022-01-12' group by 1")

# COMMAND ----------

data = spark.sql("select * from mir_dwkb.posted_transaction where post_date = '2022-01-12'")

# COMMAND ----------

data.write.mode("overwrite").parquet(tmp_path + "/skoleni/")

# COMMAND ----------

spark.sql("show create table mir_dwkb.posted_transaction").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Spark default partitioning vytvori 200 souboru

# COMMAND ----------

!hadoop fs -ls /tmp/e_rnevyh/skoleni/

# COMMAND ----------

data.coalesce(1).write.mode("overwrite").parquet(tmp_path + "/skoleni/")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pouziti coalesce(1) vytvori jeden soubor

# COMMAND ----------

!hadoop fs -ls /tmp/e_rnevyh/skoleni/

# COMMAND ----------

# MAGIC %md
# MAGIC U coalesce si je potřeba dát pozot, že query nad kterou se zavolá bude stahovat všechna data na počet serverů kolik v ní je definován.
# MAGIC V případě složitých query je nejlepší nejprve uložit výsledek bez coalesce, potom ho znovu načíst a uložit s coalesce. V takovém případě
# MAGIC neovlivníme složitost výpočtu coalescí a zároveň docílíme potřebného množství souborů.

# COMMAND ----------

data.write.mode("overwrite").parquet(tmp_path + "/skoleni_tmp/")
data2 = spark.read.parquet(tmp_path + "/skoleni_tmp/")
data.coalesce(1).write.mode("overwrite").parquet(tmp_path + "/skoleni/")

# COMMAND ----------

!hadoop fs -ls /tmp/e_rnevyh/skoleni/

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ideální počet partition
# MAGIC Většinu dat ukládáme do parquetu. Obecně nejlépe performují tabulky, jejichž podkladové parquety mají velikost kolem 120 MB.

# COMMAND ----------


