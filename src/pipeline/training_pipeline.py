from src.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_builder import CraterCNN
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluator import ModelEvaluator


def start_training_pipeline():

    config = ConfigurationManager()

    ingestion_config = config.get_data_ingestion_config()
    preprocessing_config = config.get_data_preprocessing_config()
    trainer_config = config.get_model_trainer_config()

    ingestion = DataIngestion(
        ingestion_config["dataset_download_path"]
    )

    dataset_path = ingestion.download_dataset()

    transform = DataTransformation(
        dataset_path,
        preprocessing_config["image_size"],
        preprocessing_config["batch_size"]
    )

    train_loader, val_loader, num_classes = transform.get_dataloaders()

    model = CraterCNN(num_classes)

    trainer = ModelTrainer(
        model,
        train_loader,
        val_loader,
        trainer_config["epochs"],
        trainer_config["learning_rate"],
        num_classes,
        trainer_config["model_save_path"]
    )

    trainer.train()

    evaluator = ModelEvaluator(model, val_loader)

    evaluator.evaluate()


if __name__ == "__main__":
    start_training_pipeline()
