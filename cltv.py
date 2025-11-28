from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def cltv(rfm: DataFrame, retention_rate: float = 0.7) -> DataFrame:
    return rfm.withColumn("CLTV", col("Monetary") * col("Frequency") * retention_rate)
