from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator  # Correct import for standard PythonOperator
from datetime import datetime, timedelta
import subprocess
import os

default_args = {
    "owner": "you",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Dynamically resolve full path to your script
def run_preprocessing_script():
    project_root = os.environ.get("AIRFLOW_HOME", "/usr/local/airflow")  # Fallback for Docker
    script_path = os.path.join(project_root, "src", "preprocessing.py")
    subprocess.run(['python', script_path], check=True)

with DAG(
    dag_id="stackoverflow_data_preprocessing",
    default_args=default_args,
    description="Airflow DAG to run PySpark preprocessing",
    schedule=None,
    start_date=datetime(2023, 1, 1),  # Use static start_date
    catchup=False,
    tags=["pyspark", "preprocessing"]
) as dag:
    
    preprocess_task = PythonOperator(
        task_id="run_preprocessing",
        python_callable=run_preprocessing_script
    )
