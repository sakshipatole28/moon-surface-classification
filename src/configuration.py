import yaml

class ConfigurationManager:

    def __init__(self, config_path="configs/config.yaml"):

        with open(config_path) as file:
            self.config = yaml.safe_load(file)

    def get_data_ingestion_config(self):

        return self.config["data_ingestion"]

    def get_data_preprocessing_config(self):

        return self.config["data_preprocessing"]

    def get_model_trainer_config(self):

        return self.config["model_trainer"]