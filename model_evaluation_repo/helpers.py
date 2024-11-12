import json
import os
import argparse
import numpy as np
from openai import OpenAI

from constants import QUESTION_ANSWERING, DEFAULT_BASE_MODEL, GENAI_API_BASE_URL, BASE_MODEL_TO_GENAI_MODEL_MAPPING

AUTH_TOKEN = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJGSjg2R2NGM2pUYk5MT2NvNE52WmtVQ0lVbWZZQ3FvcXRPUWVNZmJoTmxFIn0.eyJleHAiOjE3MzcxOTE3OTMsImlhdCI6MTcwNTY1NTc5MywianRpIjoiMjA4YmUxMTItYmExYS00Zjc0LWFlYjAtNzA3NTE4ODkyZjA1IiwiaXNzIjoiaHR0cDovL2dhdGV3YXkuZTJlbmV0d29ya3MuY29tL2F1dGgvcmVhbG1zL2FwaW1hbiIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiJhY2YzN2ZhOS1jZmIwLTQyNzktYjcwZi03Mjc0MDU0ZjQxZmQiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGltYW51aSIsInNlc3Npb25fc3RhdGUiOiI4NTMwYjNjMy00MWU1LTRhN2EtOGVmZi0wZGQwZDRiZjBhNjQiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImFwaXVzZXIiLCJkZWZhdWx0LXJvbGVzLWFwaW1hbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6Ijg1MzBiM2MzLTQxZTUtNGE3YS04ZWZmLTBkZDBkNGJmMGE2NCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwibmFtZSI6IkRldmFuc2ggSmFpbiIsInByaW1hcnlfZW1haWwiOiJkZXZhbnNoLmphaW5AZTJlbmV0d29ya3MuY29tIiwiaXNfcHJpbWFyeV9jb250YWN0Ijp0cnVlLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJkZXZhbnNoLmphaW5AZTJlbmV0d29ya3MuY29tIiwiZ2l2ZW5fbmFtZSI6IkRldmFuc2giLCJmYW1pbHlfbmFtZSI6IkphaW4iLCJlbWFpbCI6ImRldmFuc2guamFpbkBlMmVuZXR3b3Jrcy5jb20ifQ.egSWauw-yC93r1ceZRDEnGy5Xt-Mv-tBaHN5TBl6AJEG94GnuJvtCfWLl6FI9REGdBiIiYkByZcONiSlnporNJru82Z8D5B61uphAeJeS--GUUKNW-hSWHoR43su8hQIqLVdGJ418lF_e-t6ISj3IyLqHpWG_H3SghF6OGtsBVo'



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Argument parser for model evaluation.")

    # Dataset to be evaluated
    parser.add_argument("--dataset_type", type=str, default="eos_bucket", help="the dataset source, options: [huggingface, eos_bucket]")
    parser.add_argument("--dataset_bucket_id", type=str, default=DATASET_BUCKET_ID, help="the bucket when dataset type is eos bucket")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="the bucket path when dataset type is eos bucket")
    parser.add_argument("--dataset_accesskey", type=str, default=ACCESS_KEY, help="the bucket access key when dataset type is eos bucket")
    parser.add_argument("--dataset_secretkey", type=str, default=SECRET_KEY, help="the bucket secret key when dataset type is eos bucket")
    parser.add_argument("--hf_dataset_name", type=str, default="GAIR/o1-journey", help="the name of the dataset when dataset type is huggingface")
    parser.add_argument("--dataset_split", type=str, default="train", help="the dataset split when dataset type is huggingface")
    parser.add_argument("--num_rows_limit", type=int, default=-1, help="set the limit to number of rows for experimentation purposes")

    # Model to be used for evaluation
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL, help="default base model to use for evaluation")
    parser.add_argument("--user_auth_token", type=str, default=AUTH_TOKEN, help="the auth token of user for calling genai")
    parser.add_argument("--project_id", type=int, default=2164, help="the project id of user for calling genai")
    parser.add_argument("--temperature", type=float, default=0, help="parameter for inference used for model evaluation")
    parser.add_argument("--max_tokens", type=int, default=2048, help="parameter for inference used for model evaluation")
    parser.add_argument("--top_p", type=str, default=1, help="parameter for inference used for model evaluation")

    # Framework to be used for evaluation
    parser.add_argument("--model_evaluation_framework", type=str, default=QUESTION_ANSWERING, help="the type of model evaluation framework")
    parser.add_argument("--input_column", type=str, default="question", help="the name of the input column of the dataset")
    parser.add_argument("--output_column", type=str, default="answer", help="the name of the output column of the dataset")
    parser.add_argument("--reference_answer_column", type=str, default="actual_output", help="the name of the actual output of the dataset")
    parser.add_argument("--model_evaluation_prompt", type=str, default=None, help="model evaluation prompt given by the user")

    # Output bucket to store evaluation results
    parser.add_argument("--output_bucket_id", type=str, default=None, help="the eos bucket id to store the model evaluation results")
    parser.add_argument("--output_bucket_path", type=str, default="model_evaluation_results", help="the path in the eos bucket to store the model evaluation results")
    parser.add_argument("--output_bucket_accesskey", type=str, default=None, help="the accesskey of the output dataset bucket")
    parser.add_argument("--output_bucket_secretkey", type=str, default=None, help="the secretkey of the output dataset bucket")

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()


def is_valid_json(response):
    """
    Checks if the given response is a valid JSON format.

    Parameters:
    response (str): The response string from the model.

    Returns:
    bool: True if response is valid JSON, False otherwise.
    """
    try:
        json.loads(response)
        return True
    except (ValueError, TypeError) as e:
        print(e)
        return False


def calculate_metrics(scores):
    # Filter out invalid scores (e.g., -1)
    valid_scores = [score for score in scores if score != -1]
    invalid_count = len(scores) - len(valid_scores)

    # If no valid scores, return default metrics
    if not valid_scores:
        return {
            'mean': None,
            'median': None,
            'std_dev': None,
            'min': None,
            'max': None,
            'count': 0,
            'invalid_count': len(scores)
        }

    # Calculate metrics using numpy for valid scores
    mean = float(np.mean(valid_scores))
    median = float(np.median(valid_scores))
    std_dev = float(np.std(valid_scores))
    min_score = int(np.min(valid_scores))
    max_score = int(np.max(valid_scores))
    count = len(valid_scores)

    return {
        'mean': mean,
        'median': median,
        'std_dev': std_dev,
        'min': min_score,
        'max': max_score,
        'count': count,
        'invalid_count': invalid_count
    }


def parse_score_reason(response: str):
    try:
        # Find lines starting with "Score:" and "Reason:"
        score_line = next(line for line in response.splitlines() if line.startswith("Score:"))
        reason_line = next(line for line in response.splitlines() if line.startswith("Reason:"))

        # Extract the score and reason
        score = int(score_line.split("Score:")[1].strip())
        reason = reason_line.split("Reason:")[1].strip()

        # Ensure score is within the valid range (0-5)
        if score < 0 or score > 5:
            print(score)
            score = -1

        return score, reason
    except (ValueError, IndexError, StopIteration):
        # Handle parsing errors and return a consistent error message
        print(f"Response could not be parsed: {response}")
        return -1, response


def get_dataset_format(file_path: str):
    return os.path.splitext(file_path)[1]


def get_genai_client(base_model_id: str, project_id: int, api_key: str):
    genai_model_id = BASE_MODEL_TO_GENAI_MODEL_MAPPING.get(base_model_id)
    client = OpenAI(
        base_url=GENAI_API_BASE_URL.format(project_id=project_id, genai_model_id=genai_model_id),
        api_key=api_key
    )
    return client
