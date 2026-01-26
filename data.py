import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Dataset downloaded to:", path)
