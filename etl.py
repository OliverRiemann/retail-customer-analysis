from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.functions import max, datediff, sum as _sum, count as _count
from pyspark.sql.functions import lit

spark = SparkSession.builder.getOrCreate()

def load_data(input_path: str) -> DataFrame:
    return spark.read.csv(input_path, header=True, inferSchema=True)

def preprocess_data(df: DataFrame) -> DataFrame:
    df_clean = df.dropna(subset=["InvoiceNo", "CustomerID", "InvoiceDate", "UnitPrice", "Quantity"])

    df_clean = df_clean.filter(col("Quantity") > 0)

    df_clean = df_clean.dropDuplicates()

    return df_clean

def rfm_calculation(df: DataFrame) -> DataFrame:
    snapshot_date = df.agg(max("InvoiceDate")).collect()[0][0]

    rfm = df.groupBy("CustomerID").agg(
        datediff(lit(snapshot_date), max("InvoiceDate")).alias("Recency"),
        _count("InvoiceNo").alias("Frequency"),
        _sum(col("UnitPrice") * col("Quantity")).alias("Monetary")
    )

    return rfm

df = load_data("online_retail.csv")
df_clean = preprocess_data(df)
rfm = rfm_calculation(df_clean)
