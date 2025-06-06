import os
import inspect
import subprocess
from typing import Any, Iterable, Generator

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import RepoFile, RepoFolder, ModelInfo

import transformers
from transformers.models.auto import (
    configuration_auto,
    feature_extraction_auto,
    image_processing_auto,
    modeling_auto,
    # modeling_flax_auto,
    # modeling_tf_auto,
    processing_auto,
    tokenization_auto,
)

# BEGIN: patched downloads
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import CLIPFeatureExtractor

from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from transformers import RobertaConfig
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification

from transformers import XLMRobertaConfig
from transformers import XLMRobertaTokenizer
from transformers import XLMRobertaForSequenceClassification

from peft import AutoPeftModelForCausalLM

from sentence_transformers import SentenceTransformer
# END: patched downloads

from ch import constants as ch_cst
from ch.state import JOB_META

from validator.utils.logging import get_logger

logger = get_logger(__name__)

_HF_MAPPING_NAMES = (
    configuration_auto.CONFIG_MAPPING_NAMES,
    feature_extraction_auto.FEATURE_EXTRACTOR_MAPPING_NAMES,
    image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES,
    modeling_auto.MODEL_MAPPING_NAMES,
    # modeling_flax_auto.FLAX_MODEL_MAPPING_NAMES,
    # modeling_tf_auto.TF_MODEL_MAPPING_NAMES,
    processing_auto.PROCESSOR_MAPPING_NAMES,
    tokenization_auto.TOKENIZER_MAPPING_NAMES,
)

_HF_SKIP_MODELS = {
    "Bamba",
    "BambaModel",
    "BambaConfig",
    "BambaForCausalLM",
    "Jamba",
    "JambaModel",
    "JambaForCausalLM",
    "JambaForSequenceClassification",
    "Mamba",
    "MambaModel",
    "MambaConfig",
    "MambaForCausalLM",
    "Mamba2Model",
    "Mamba2Config",
    "Mamba2ForCausalLM",
    "FalconMambaModel",
    "Zamba",
    "ZambaModel",
    "ZambaConfig",
    "ZambaForCausalLM",
    "ZambaForSequenceClassification",
    "Zamba2",
    "Zamba2Model",
    "Zamba2Config",
    "Zamba2ForCausalLM",
    "Zamba2ForSequenceClassification",
    "Pop2PianoProcessor",
    "Pop2PianoFeatureExtractor",
    "TimmWrapperConfig",
    "TimmWrapperModel",
    "TimmWrapperImageProcessor",
    "TimmWrapperForImageClassification",
}

_HF_API = HfApi()
_PATCHED = False


class LoggingHfApi:
    def list_repo_files(self, repo_id: str) -> list[str]:
        logger.info(f"*** LoggingHfApi.list_repo_files: {repo_id}")
        files = _HF_API.list_repo_files(repo_id=repo_id)
        for file in files:
            logger.info(f"*** {file}")
        return files

    def list_repo_tree(self, repo_id: str, repo_type="model") -> Iterable[RepoFile | RepoFolder]:
        logger.info(f"*** LoggingHfApi.list_repo_tree: [{repo_type}] {repo_id}")
        entries = _HF_API.list_repo_tree(repo_id=repo_id, repo_type=repo_type)
        for entry in entries:
            if hasattr(entry, "size") and entry.size is not None:
                logger.info(f"*** {entry.path} {entry.size} bytes")
        return entries

    def model_info(self, repo_id: str) -> ModelInfo:
        logger.info(f"*** LoggingHfApi.model_info: {repo_id}")
        model_info = _HF_API.model_info(repo_id=repo_id)
        if model_info and model_info.lastModified:
            logger.info(f"*** {model_info.lastModified=}")
        return model_info


class InputVolumeHfApi:
    def list_repo_files(self, repo_id: str) -> list[str]:
        logger.info(f"*** InputVolumeHfApi.list_repo_files: {repo_id}")
        files = []
        filename = JOB_META["hf"][repo_id]
        if filename is not None:
            files.append(filename)
        for file in files:
            logger.info(f"*** {file}")
        return files

    def list_repo_tree(self, repo_id: str, repo_type="model") -> Iterable[RepoFile | RepoFolder]:
        logger.info(f"*** InputVolumeHfApi.list_repo_tree: [{repo_type}] {repo_id}")
        assert repo_type == "model"
        filename = JOB_META["hf"][repo_id]
        if filename is None:
            return []
        entries = [RepoFile(path=filename, size=ch_cst.HF_LIST_REPO_FAKE_FILE_SIZE, oid="id")]
        for entry in entries:
            if hasattr(entry, "size") and entry.size is not None:
                logger.info(f"*** {entry.path} {entry.size} bytes")
        return entries

    def model_info(self, repo_id: str) -> ModelInfo:
        logger.info(f"*** InputVolumeHfApi.model_info: {repo_id}")
        model_info = ModelInfo(
            modelId=repo_id,
            lastModified=None,
        )
        return model_info


def logging_snapshot_download(*, repo_id: str, local_dir: str, repo_type: str) -> str:
    logger.info(f"*** logging_snapshot_download: [{repo_type}] {repo_id} {local_dir}")
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type=repo_type)
    logger.info(f"*** {path}")
    return path


def logging_hf_hub_download(*, repo_id: str, filename: str, local_dir: str) -> str:
    logger.info(f"*** logging_hf_hub_download: {repo_id} {filename} {local_dir}")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    logger.info(f"*** {path}")
    return path


def input_volume_snapshot_download(*, repo_id: str, local_dir: str, repo_type: str) -> str:
    logger.info(f"*** input_volume_snapshot_download: [{repo_type}] {repo_id} {local_dir}")
    assert repo_id in JOB_META["hf"]
    path = os.path.abspath(local_dir)
    source_path = f"/volume/hf/{repo_id}"
    logger.info(f"+++ Creating symlink from {source_path} to {path}")
    result = subprocess.run(
        ["ln", "-s", source_path, path],
        capture_output=True,
        text=True,
    )
    logger.info("\n" + result.stdout.strip())
    logger.info("\n" + result.stderr.strip())
    assert result.returncode == 0
    logger.info(f"*** {source_path}")
    return source_path


def input_volume_hf_hub_download(*, repo_id: str, filename: str, local_dir: str) -> str:
    logger.info(f"*** input_volume_hf_hub_download: {repo_id} {filename} {local_dir}")
    assert JOB_META["hf"][repo_id] == filename
    path = os.path.abspath(os.path.join(local_dir, filename))
    source_path = f"/volume/hf/{repo_id}/{filename}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"+++ Creating symlink from {source_path} to {path}")
    result = subprocess.run(
        ["ln", "-s", source_path, path],
        capture_output=True,
        text=True,
    )
    logger.info("\n" + result.stdout.strip())
    logger.info("\n" + result.stderr.strip())
    assert result.returncode == 0
    logger.info(f"*** {path}")
    return path


_original_SentenceTransformer__init__ = SentenceTransformer.__init__


def _logging_SentenceTransformer__init__(self, model_name_or_path: str, *args, **kwargs):
    logger.info(f"*** logging SentenceTransformer.__init__: {model_name_or_path}")
    return _original_SentenceTransformer__init__(self, model_name_or_path, *args, **kwargs)


def _input_volume_SentenceTransformer__init__(self, model_name_or_path: str, *args, **kwargs):
    logger.info(f"*** input_volume SentenceTransformer.__init__: {model_name_or_path}")
    if model_name_or_path.startswith("/"):
        source_path = model_name_or_path
    else:
        assert model_name_or_path.count("/") == 1
        source_path = f"/volume/hf/{model_name_or_path}"
    return _original_SentenceTransformer__init__(self, source_path, *args, **kwargs)


def _patch_hf_model(model_cls: Any) -> None:
    model_name = model_cls.__name__
    original_from_pretrained = model_cls.from_pretrained

    @classmethod
    def logging_from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        logger.info(f"*** logging {model_name}.from_pretrained: {pretrained_model_name_or_path}")
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    @classmethod
    def input_volume_from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        logger.info(f"*** input_volume {model_name}.from_pretrained: REQ {pretrained_model_name_or_path}")
        if pretrained_model_name_or_path is None or pretrained_model_name_or_path.startswith("/"):
            source_path = pretrained_model_name_or_path
        else:
            if "/" not in pretrained_model_name_or_path:
                assert pretrained_model_name_or_path in ch_cst.FULL_MODEL_NAME
                pretrained_model_name_or_path = ch_cst.FULL_MODEL_NAME[pretrained_model_name_or_path]
            assert pretrained_model_name_or_path.count("/") == 1
            source_path = f"/volume/hf/{pretrained_model_name_or_path}"
        logger.info(f"*** input_volume {model_name}.from_pretrained: USE {source_path}")
        return original_from_pretrained(source_path, *args, **kwargs)

    # check if the method was already patched
    if model_cls.from_pretrained.__func__.__module__ != "ch.download":
        # logger.info(f"*** PATCHING: {model_name}.from_pretrained")
        if ch_cst.USE_INPUT_VOLUME:
            model_cls.from_pretrained = input_volume_from_pretrained
        else:
            model_cls.from_pretrained = logging_from_pretrained


def _iter_hf_model_classes(mapping: dict) -> Generator[type, None, None]:
    for model_classes in mapping.values():
        if isinstance(model_classes, str):
            model_classes = [model_classes]
        for model_class in model_classes:
            if model_class is None:
                continue
            if model_class in _HF_SKIP_MODELS:
                continue

            logger.info(f"+++ HF class: {model_class}")
            cls = getattr(transformers, model_class, None)
            if cls is None:
                continue

            method = getattr(cls, "from_pretrained", None)
            if method is None:
                continue

            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            if not params:
                continue
            first_param = params[0]
            if first_param != "pretrained_model_name_or_path":
                continue

            yield cls


def _get_hf_all_model_classes() -> set[type]:
    classes = set()
    for mapping_names in _HF_MAPPING_NAMES:
        for cls in _iter_hf_model_classes(mapping_names):
            classes.add(cls)
    return classes


def patch() -> None:
    global _PATCHED
    if not ch_cst.USE_PATCHING or _PATCHED:
        return
    _PATCHED = True

    logger.info("+++ Download patching")

    for cls in _get_hf_all_model_classes():
        _patch_hf_model(cls)

    _patch_hf_model(AutoConfig)
    _patch_hf_model(AutoTokenizer)
    _patch_hf_model(AutoModelForCausalLM)
    _patch_hf_model(AutoModelForSequenceClassification)
    _patch_hf_model(CLIPFeatureExtractor)

    _patch_hf_model(BertConfig)
    _patch_hf_model(BertTokenizer)
    _patch_hf_model(BertForSequenceClassification)

    _patch_hf_model(RobertaConfig)
    _patch_hf_model(RobertaTokenizer)
    _patch_hf_model(RobertaForSequenceClassification)

    _patch_hf_model(XLMRobertaConfig)
    _patch_hf_model(XLMRobertaTokenizer)
    _patch_hf_model(XLMRobertaForSequenceClassification)

    _patch_hf_model(AutoPeftModelForCausalLM)

    if ch_cst.USE_INPUT_VOLUME:
        SentenceTransformer.__init__ = _input_volume_SentenceTransformer__init__
    else:
        SentenceTransformer.__init__ = _logging_SentenceTransformer__init__
