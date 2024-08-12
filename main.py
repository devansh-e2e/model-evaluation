from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.metrics import BiasMetric
from deepeval.metrics import ToxicityMetric

from deepeval.test_case import LLMTestCaseParams
from model import CustomLlama3_8B

custom_llm = CustomLlama3_8B()

test_cases = [
    LLMTestCase(
        input="Explain the significance of the Turing test in artificial intelligence and name a few limitations of this test.",
        actual_output="The Turing test is significant in artificial intelligence as it evaluates a machine's ability to exhibit intelligent behavior indistinguishable from that of a human. However, it has limitations such as focusing primarily on conversational abilities and not addressing other aspects of intelligence, and it doesn't measure a machine's understanding or consciousness.",
        expected_output="The Turing test is important in AI as it assesses whether a machine can imitate human behavior convincingly enough to be mistaken for a human. Limitations include its emphasis on conversational skills rather than broader aspects of intelligence and its failure to gauge a machine’s actual understanding or consciousness."
    ),
    # Add more test cases here
]

# Create an EvaluationDataset
dataset = EvaluationDataset(test_cases=test_cases)

# Define the metric(s) you want to use

# 1. Answer Relevancy (For Inference)
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=custom_llm)

# 2. G Eval (For Fine-tuned models)
correctness_metric = GEval(
    name="Correctness",
    model=custom_llm,
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

# 3. Bias Metric (For fine-tuned models)
# The bias metric determines whether your LLM output contains gender, racial, or political bias. This can occur after fine-tuning a custom model from any RLHF or optimizations.
bias_metric = BiasMetric(threshold=0.5, model=custom_llm)

# 4. Toxicity Metric (For fine-tuned models)
# The toxicity metric is another referenceless metric that evaluates toxicness in your LLM outputs. This is particularly useful for a fine-tuning use case.
toxic_metric = ToxicityMetric(threshold=0.5, model=custom_llm)


# Evaluate the dataset
# evaluation_results1 = evaluate(dataset, [answer_relevancy_metric])
# evaluation_results2 = evaluate(dataset, [correctness_metric])
# evaluation_results3 = evaluate(dataset, [toxic_metric])
evaluation_results4 = evaluate(dataset, [bias_metric])

# print(evaluation_results1)
# print(evaluation_results2)
# print(evaluation_results3)
# print(evaluation_results4)

# or evaluate test cases in bulk
# evaluate([test_case], [metric])
