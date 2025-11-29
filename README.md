# 🛍️ Customer Lifetime Value (CLTV) & Churn Prediction Pipeline with Databricks

This project demonstrates an end-to-end **data engineering and machine learning pipeline** using **Databricks**, with a focus on:

- Calculating **Customer Lifetime Value (CLTV)**
- Predicting **customer churn**
- Visualizing key customer insights using **SQL Dashboards**

> ✅ Built to showcase my portfolio with best practices in PySpark, Delta Lake, MLflow, Unity Catalog, and Databricks Jobs.

---

## 🎯 Objective

To identify high-value customers and predict churn likelihood using e-commerce transaction data — enabling better **marketing targeting**, **retention planning**, and **customer segmentation**.

---

## 🧱 Pipeline Architecture

               ┌──────────────────┐
               │  CSV Source File │
               └────────┬─────────┘
                        ▼
               ┌──────────────────┐
               │   ETL Notebook   │
               │ - Clean & filter │
               │ - RFM metrics    │
               └────────┬─────────┘
                        ▼
               ┌────────────────────┐
               │ Churn Model        │
               │ - Logistic Regression
               │ - MLflow logging   │
               └────────┬───────────┘
                        ▼
               ┌──────────────────┐
               │   CLTV Notebook  │
               │ - Calculate CLTV │
               └────────┬─────────┘
                        ▼
               ┌────────────────────────────┐
               │ Databricks SQL Dashboard   │
               └────────────────────────────┘

