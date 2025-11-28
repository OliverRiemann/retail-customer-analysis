from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame

# Create churn label (example: customers with no purchases in last N days)
def build_rfm(data: DataFrame):
    rfm = data.withColumn("Churn", (data["Recency"] > 90).cast("integer"))

    features = ["Recency", "Frequency", "Monetary"]
    assembler = VectorAssembler(inputCols=features, outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol="Churn")

    pipeline = Pipeline(stages=[assembler, lr])

    train, test = rfm.randomSplit([0.8, 0.2])
    model = pipeline.fit(train)

    predictions = model.transform(test)
    predictions.select("Churn", "prediction", "probability").show(10)

    return rfm

