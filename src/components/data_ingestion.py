import os
import sys
from roboflow import Roboflow
from dotenv import load_dotenv

from src.logger import logger
from src.exception import CustomException


class DataIngestion:

    def __init__(self, download_path):
        self.download_path = download_path

    def download_dataset(self):

        try:

            logger.info("Starting dataset download")

            # Load environment variables
            load_dotenv()

            api_key = os.getenv("ROBOFLOW_API_KEY")

            if api_key is None:
                raise ValueError("Roboflow API key not found in .env")

            rf = Roboflow(api_key=api_key)

            project = rf.workspace("sakshis-workspace-rjo48").project("hons-lunar-ai-skhp1")

            version = project.version(1)

            dataset = version.download("multiclass")

            logger.info(f"Dataset downloaded at {dataset.location}")

            return dataset.location

        except Exception as e:
            raise CustomException(e, sys)