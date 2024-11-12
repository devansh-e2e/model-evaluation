from deepeval.benchmarks import BigBenchHard
from deepeval.benchmarks.tasks import BigBenchHardTask

from models.model import CustomInferenceModel

custom_llm = CustomInferenceModel()
# Define benchmark with specific tasks and shots
benchmark = BigBenchHard(
    tasks=[BigBenchHardTask.BOOLEAN_EXPRESSIONS, BigBenchHardTask.CAUSAL_JUDGEMENT],
    n_shots=3,
    enable_cot=True
)

# Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=custom_llm)
print(benchmark.overall_score)
