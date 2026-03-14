from src.pipeline.prediction_pipeline import PredictionPipeline

predictor = PredictionPipeline("artifacts/model.pth", num_classes=9)

print(predictor.predict("HONS-Lunar-AI-1/test/Lunarmount010_jpg.rf.b1414ffa5ca71090c01e63dc726e155e.jpg"))