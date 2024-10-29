from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from models.model import CustomInferenceModel

custom_llm = CustomInferenceModel()

# Define benchmark with specific tasks and shots
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=3
)

# Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=custom_llm)
print(benchmark.overall_score)

# Below is the comprehensive list of all available tasks:

# HIGH_SCHOOL_EUROPEAN_HISTORY
# BUSINESS_ETHICS
# CLINICAL_KNOWLEDGE
# MEDICAL_GENETICS
# HIGH_SCHOOL_US_HISTORY
# HIGH_SCHOOL_PHYSICS
# HIGH_SCHOOL_WORLD_HISTORY
# VIROLOGY
# HIGH_SCHOOL_MICROECONOMICS
# ECONOMETRICS
# COLLEGE_COMPUTER_SCIENCE
# HIGH_SCHOOL_BIOLOGY
# ABSTRACT_ALGEBRA
# PROFESSIONAL_ACCOUNTING
# PHILOSOPHY
# PROFESSIONAL_MEDICINE
# NUTRITION
# GLOBAL_FACTS
# MACHINE_LEARNING
# SECURITY_STUDIES
# PUBLIC_RELATIONS
# PROFESSIONAL_PSYCHOLOGY
# PREHISTORY
# ANATOMY
# HUMAN_SEXUALITY
# COLLEGE_MEDICINE
# HIGH_SCHOOL_GOVERNMENT_AND_POLITICS
# COLLEGE_CHEMISTRY
# LOGICAL_FALLACIES
# HIGH_SCHOOL_GEOGRAPHY
# ELEMENTARY_MATHEMATICS
# HUMAN_AGING
# COLLEGE_MATHEMATICS
# HIGH_SCHOOL_PSYCHOLOGY
# FORMAL_LOGIC
# HIGH_SCHOOL_STATISTICS
# INTERNATIONAL_LAW
# HIGH_SCHOOL_MATHEMATICS
# HIGH_SCHOOL_COMPUTER_SCIENCE
# CONCEPTUAL_PHYSICS
# MISCELLANEOUS
# HIGH_SCHOOL_CHEMISTRY
# MARKETING
# PROFESSIONAL_LAW
# MANAGEMENT
# COLLEGE_PHYSICS
# JURISPRUDENCE
# WORLD_RELIGIONS
# SOCIOLOGY
# US_FOREIGN_POLICY
# HIGH_SCHOOL_MACROECONOMICS
# COMPUTER_SECURITY
# MORAL_SCENARIOS
# MORAL_DISPUTES
# ELECTRICAL_ENGINEERING
# ASTRONOMY
# COLLEGE_BIOLOGY