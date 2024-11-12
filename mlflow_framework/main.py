import mlflow
from models.model import CustomInferenceModel
from dataset_loader import CustomDatasetLoader

# User-provided dataset information
DATASET_NAME = "Malikeh1375/medical-question-answering-datasets"  # Example dataset
INPUT_FIELD = "input"
GROUND_TRUTH_FIELD = "output"

# Load dataset using DatasetLoader
dataset_loader = CustomDatasetLoader(DATASET_NAME, INPUT_FIELD, GROUND_TRUTH_FIELD)
eval_df = dataset_loader.load_dataset()

# Define sample input and inference parameters
sample_input = "What is machine learning?"
inference_params = {
    "max_new_tokens": 300,
    "do_sample": True,
}

# Initialize and log the model with MLflow
model = CustomInferenceModel(model_id="prometheus-eval/prometheus-8x7b-v2.0")
llama_pipeline = model.get_pipeline()

# Log model with MLflow and evaluate it
with mlflow.start_run() as run:
    # Log the model
    signature_output = llama_pipeline(sample_input, **inference_params)[0]["generated_text"]
    signature = mlflow.models.infer_signature(sample_input, signature_output)

    model_info = mlflow.transformers.log_model(
        transformers_model=llama_pipeline,
        artifact_path="model",
        task="text-generation",
        signature=signature,
        input_example="A clever and witty question",
    )

    # Evaluate the model
    results = mlflow.evaluate(
        model_info.model_uri,
        eval_df,
        targets="ground_truth",
        model_type="text-summarization",
        evaluators="default"
    )

# Print evaluation metrics
print("Evaluation Metrics:", results.metrics)
