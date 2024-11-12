import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline


class CustomInferenceModel:
    def __init__(self, model_id='prometheus-eval/prometheus-8x7b-v2.0'):
        """
        Initialize the inference model with specified model ID and quantization configuration.

        Parameters:
            model_id (str): The Hugging Face model ID for the causal language model.
        """
        self.model_id = model_id
        self.quantization_config = self._set_quantization_config()
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _set_quantization_config(self):
        """
        Define and return the quantization configuration for model loading.

        Returns:
            BitsAndBytesConfig: Configuration for 4-bit quantization.
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
        )

    def _load_model_and_tokenizer(self):
        """
        Load the model and tokenizer from Hugging Face with quantization.

        Returns:
            Tuple: Loaded model and tokenizer objects.
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=self.quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return model, tokenizer

    def load_model(self):
        """
        Load and return the model for inference.

        Returns:
            PreTrainedModel: The loaded model for inference.
        """
        return self.model

    def get_model_name(self):
        """
        Return the model ID name.

        Returns:
            str: Model ID.
        """
        return self.model_id

    def get_pipeline(self):
        """
        Create and return a text generation pipeline using the loaded model and tokenizer.

        Returns:
            Pipeline: Text generation pipeline for model inference.
        """
        return pipeline(
            "text-generation",
            model=self.load_model(),
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
