import os

RUNNING_IN_CONTAINER = os.path.isdir("/aplp")

USE_PATCHING = True
USE_INPUT_VOLUME = True
USE_RANDOM_SEED = False

CH_VALIDATION_DUAL = False
CH_VALIDATION_DPO_TASK = True
CH_VALIDATION_GRPO_TASK = True
CH_VALIDATION_IMAGE_TASK = True
CH_VALIDATION_INSTRUCT_TEXT_TASK = True

CH_USE_PROD_ENV = True
if CH_USE_PROD_ENV:
    CH_JOB_FACILITATOR_URL = "https://facilitator.computehorde.io/"
    CH_JOB_TARGET_VALIDATOR_HOTKEY = "5HBVrFGy6oYhhh71m9fFGYD7zbKyAeHnWN8i8s9fJTBMCtEE"
else:
    CH_JOB_FACILITATOR_URL = "https://testing.facilitator.computehorde.io/"
    CH_JOB_TARGET_VALIDATOR_HOTKEY = "5GKUbKHhgonqJKoaSSNnXxL1Gz1sWSg4ekS7SLZ3ACBTEMMt"

CH_JOB_NAMESPACE = "SN56.0"
CH_JOB_SIGNER_WALLET_NAME = "_validator"
CH_JOB_SIGNER_WALLET_HOTKEY_NAME = "default"

CH_JOB_VALIDATOR_DOCKER_IMAGE = "backenddevelopersltd/ch-sn56-tuning_vali"
CH_JOB_VALIDATOR_DOCKER_IMAGE_DIFFUSION = "backenddevelopersltd/ch-sn56-tuning_validator_diffusion"

CH_LOG_DIR = os.path.expanduser("~/.sn56")

# replace model name with full mounted input volume path for use in pretrained_model_name_or_path
# used by `from_pretrained(cls, pretrained_model_name_or_path: str, ...)`
# "google-bert/bert-base-uncased" -> "/volume/hf/google-bert/bert-base-uncased"
HF_REPLACE_NAME_WITH_PATH = False

HF_LIST_REPO_FAKE_FILE_SIZE = 11 * 1024 * 1024 * 1024  # 11 GB

FULL_MODEL_NAME = {
    "bert-base-uncased": "google-bert/bert-base-uncased",
    "roberta-base": "FacebookAI/roberta-base",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
}

# https://github.com/unitaryai/detoxify#prediction
DETOXIFY_MODEL = {
    "original": "google-bert/bert-base-uncased",
    "unbiased": "FacebookAI/roberta-base",
    "multilingual": "FacebookAI/xlm-roberta-base",
}

# https://github.com/citadel-ai/langcheck/blob/main/src/langcheck/metrics/model_manager/config/metric_config.yaml
LANGCHECK_EN_MODEL = {
    "semantic_similarity": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "model_revision": "c46f31f8d119ac648208de9fba297c447a5a4474",
    },
    "fluency": {
        "model_name": "prithivida/parrot_fluency_model",
        "model_revision": "e5224ff5b4109cd949ce25b0a6dff8d8cbdec7be",
    },
    "sentiment": {
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "model_revision": "4ba3d4463bd152c9e4abd892b50844f30c646708",
    },
}

#
# PATCH
#

import logging

import transformers.utils.hub
import transformers.modeling_utils

logger = logging.getLogger(__name__)

_original_transformers_hub_has_file = transformers.utils.hub.has_file


def _logging_transformers_hub_has_file(
    path_or_repo: str | os.PathLike,
    filename: str,
    *args,
    **kwargs,
) -> bool:
    logger.info(f"*** logging transformers.utils.hub.has_file: {path_or_repo=} {filename=}")
    result = _original_transformers_hub_has_file(path_or_repo, filename, *args, **kwargs)
    logger.info(f"*** logging transformers.utils.hub.has_file: {result=}")
    return result


transformers.utils.hub.has_file = _logging_transformers_hub_has_file
transformers.modeling_utils.has_file = _logging_transformers_hub_has_file
