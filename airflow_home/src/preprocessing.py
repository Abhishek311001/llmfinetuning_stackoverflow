from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit
import subprocess
import os
import logging

import logging

import logging
import sys

# Configure logger to use stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Create handler that logs to stdout (not stderr)
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)



def preprocess_and_versiondata():
    logger.info("Collecting paths")
    project_root = os.environ.get("AIRFLOW_HOME", "/usr/local/airflow")  # fallback if not set
    repo_root = "/home/azureuser/cloudfiles/code/Users/abhishekbatti2001/llmfinetuning_stackoverflow"
    data_root = os.path.join(repo_root, "data")
    raw_data_path = os.path.join(data_root, "raw", "stackoverflow.json")
    processed_dir = os.path.join(data_root, "preprocessed")
    processed_file = os.path.join(processed_dir, "preprocessed_data.parquet")
    logger.info("paths Collected")
    logger.info("Creating spark app")
    # Create Spark session
    spark = SparkSession.builder.appName("StackoverflowDataIngestion").getOrCreate()
    logger.info("Created")
    # Load and transform data

    logger.info("Ingesting Data")
    df_raw = spark.read.json(raw_data_path)
    logger.info("Data Ingested")
    logger.info("Preprocssing started....")
    df_raw = df_raw.select(
        df_raw.answer_body.alias("Answer"),
        df_raw.question_body,
        df_raw.question_title,
        df_raw.score
    )
    df = df_raw.withColumn(
        "Question",
        concat(
            lit("**Question Title** \n"), col("question_title"),
            lit("\n \n **Question Body**\n"), col("question_body")
        )
    ).select("Answer", "Question", "score")
    logger.info("Preprocssing Finished!!")
    # Write processed data
    logger.info("writing preprocessed file...")
    df.write.mode("overwrite").parquet(processed_file)
    logger.info("Finished writing preprocessed file")
    # DVC commands
    dvc_target_path = os.path.join("data", "preprocessed", "preprocessed_data.parquet")
    logger.info("Executing DVC Commands")
    subprocess.run(["dvc", "add", dvc_target_path], check=True, cwd=repo_root)
    subprocess.run(["git", "add", f"{dvc_target_path}.dvc"], check=True, cwd=repo_root)
    subprocess.run(["git", "commit", "-m", "Add processed stackoverflow data"], check=True, cwd=repo_root)
    subprocess.run(["dvc", "push"], check=True, cwd=repo_root)

    logger.info("Function Complete")

if __name__ == "__main__":
    preprocess_and_versiondata()
