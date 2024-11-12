from datasets import load_dataset
import logging
import sys

from constants import EOS_BUCKET, ALLOWED_FILE_TYPES
from e2enetworks.cloud.tir import Datasets
from helpers import get_dataset_format

logger = logging.getLogger(__name__)


class DatasetLoader():
    def __init__(self, args):

        self.dataset_type = args.dataset_type
        self.dataset_path = args.dataset_path
        self.dataset_bucket_id = args.dataset_bucket_id
        self.hf_dataset_name = args.hf_dataset_name
        self.dataset_split = args.dataset_split
        self.num_rows_limit = args.num_rows_limit

        self.dataset = Datasets()

    def load(self):
        if self.dataset_type != EOS_BUCKET:
            return self._load_huggingface_dataset()
        logger.info(f"Loading custom dataset from {self.dataset_path}")
        dataset = self._load_custom_dataset()
        if self.num_rows_limit > 0:
            dataset = dataset.select(range(self.num_rows_limit))

    def _load_custom_dataset(self):
        file_type = get_dataset_format(self.dataset_path)
        if file_type not in ALLOWED_FILE_TYPES:
            logger.error(f"File type '{file_type}' is not supported. Allowed types are: {ALLOWED_FILE_TYPES}")
            sys.exit(f"Unsupported file type: {file_type}")

        try:
            return self.dataset.load_dataset_file(dataset_id=self.dataset_bucket_id, file_path=self.dataset_path)
        except Exception as e:
            logger.error(f"ERROR_IN_LOADING_DATASET: {e}")
            sys.exit(e)

    def _load_huggingface_dataset(self):
        logger.info(f"Loading dataset {self.dataset_name} from huggingface")
        return load_dataset(self.hf_dataset_name, split=self.dataset_split)
