from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Create churn label (example: customers with no purchases in last N days)
def lr_model(rfm_data: DataFrame, churn_recency: int = 90, model_path: str = "models/churn_lr"):
    rfm = rfm_data.withColumn("Churn", (rfm_data["Recency"] > churn_recency).cast("integer"))

    features = ["Recency", "Frequency", "Monetary"]
    assembler = VectorAssembler(inputCols=features, outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol="Churn")

    pipeline = Pipeline(stages=[assembler, lr])

    train, test = rfm.randomSplit([0.8, 0.2])
    model = pipeline.fit(train)

    model.write().overwrite().save(model_path)

     # Evaluate
    preds = model.transform(test)
    roc_auc = BinaryClassificationEvaluator(labelCol="Churn", metricName="areaUnderROC").evaluate(preds)
    pr_auc = BinaryClassificationEvaluator(labelCol="Churn", metricName="areaUnderPR").evaluate(preds)
    accuracy = MulticlassClassificationEvaluator(labelCol="Churn", metricName="accuracy").evaluate(preds)
    print(f"ROC AUC: {roc_auc}, PR AUC: {pr_auc}, Accuracy: {accuracy}")

    return model


