from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import max, datediff, sum as _sum, count as _count
from pyspark.sql.functions import lit

spark = SparkSession.builder.getOrCreate()



def load_transform(data: str):
    df = spark.read.csv(data, header=True, inferSchema=True)
 
    df_clean = df.dropna(subset=["InvoiceNo", "CustomerID", "InvoiceDate", "UnitPrice", "Quantity"])

    df_clean = df_clean.filter(col("Quantity") > 0)

    df_clean = df_clean.dropDuplicates()

    snapshot_date = df_clean.agg(max("InvoiceDate")).collect()[0][0]

    rfm = df_clean.groupBy("CustomerID").agg(
    datediff(lit(snapshot_date), max("InvoiceDate")).alias("Recency"),
    _count("InvoiceNo").alias("Frequency"),
    _sum(col("UnitPrice") * col("Quantity")).alias("Monetary")
    )

    return rfm
    
load_transform('online_retail.csv').show()
