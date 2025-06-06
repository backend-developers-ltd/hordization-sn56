import os
import re
import json
import pprint
import asyncio
from typing import Any
from datetime import datetime
from dataclasses import asdict
from urllib.parse import urlparse

import bittensor_wallet
from detoxify import detoxify
from compute_horde_sdk.v1 import (
    ComputeHordeClient,
    ComputeHordeJobSpec,
    ExecutorClass,
    ComputeHordeJobStatus,
    InputVolume,
    InlineInputVolume,
    HTTPInputVolume,
    HuggingfaceInputVolume,
)

from ch import constants as ch_cst

from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from validator.utils.logging import get_logger
from validator.evaluation.eval_diffusion import is_safetensors_available, find_latest_lora_submission_name

logger = get_logger(__name__)


def log_run_evaluation_docker_image(
    test_split_url: str,
    original_model_repo: str,
    models: list[str],
    model_type: ImageModelType,
    gpu_ids: list[int],
    container_dataset_path: str,
    log_id: str | None = None,
    eval_results: dict | None = None,
) -> str:
    os.makedirs(ch_cst.CH_LOG_DIR, exist_ok=True)

    if log_id is None:
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S-image")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "test_split_url": test_split_url,
        "original_model_repo": original_model_repo,
        "models": models,
        "model_type": model_type.value,
        "gpu_ids": gpu_ids,
        "container_dataset_path": container_dataset_path,
        "eval_results": eval_results,
    }

    log_path = os.path.join(ch_cst.CH_LOG_DIR, f"{log_id}.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, separators=(",", ": "))

    return log_id


def log_run_evaluation_docker_text(
    dataset_url: str,
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType,
    file_format: FileFormat,
    gpu_ids: list[int],
    command: list[str],
    environment: dict[str, str],
    log_id: str | None = None,
    eval_results: dict | None = None,
) -> str:
    os.makedirs(ch_cst.CH_LOG_DIR, exist_ok=True)

    if log_id is None:
        log_id = datetime.now().strftime("%Y%m%d_%H%M%S-text")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset_url": dataset_url,
        "dataset": dataset,
        "models": models,
        "original_model": original_model,
        "dataset_type": dataset_type.model_dump(),
        "file_format": file_format.value,
        "gpu_ids": gpu_ids,
        "command": command,
        "environment": environment,
        "eval_results": eval_results,
    }

    log_path = os.path.join(ch_cst.CH_LOG_DIR, f"{log_id}.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, separators=(",", ": "))

    return log_id


async def run_evaluation_docker_image(
    test_split_url: str,
    original_model_repo: str,
    models: list[str],
    model_type: ImageModelType,
    gpu_ids: list[int],
    container_dataset_path: str,
) -> dict:
    logger.info("-=+*" * 20)
    logger.info("----- run_evaluation_docker_image -----")
    logger.info(f"{test_split_url=}")
    logger.info(f"{original_model_repo=}")
    logger.info(f"{models=}")
    logger.info(f"{model_type=}")
    logger.info(f"{gpu_ids=}")
    logger.info("-=+*" * 20)

    job_obj: dict[str, Any] = dict(
        hf={},
    )

    job_obj["env"] = {
        "DATASET": container_dataset_path,
        "MODELS": ",".join(models),
        "ORIGINAL_MODEL_REPO": original_model_repo,
        "MODEL_TYPE": model_type.value,
    }

    input_volumes: dict[str, InputVolume] = {}

    logger.info(f">>>> {original_model_repo}")
    _is_safetensors, safetensors_filename = is_safetensors_available(original_model_repo, model_type)
    job_obj["hf"][original_model_repo] = safetensors_filename
    input_volumes[f"/volume/hf/{original_model_repo}"] = HuggingfaceInputVolume(
        repo_id=original_model_repo,
        repo_type="model",
        allow_patterns=[safetensors_filename] if safetensors_filename else None,
    )

    for repo_id in models:
        logger.info(f">>>> {repo_id}")
        lora_filename = find_latest_lora_submission_name(repo_id)
        job_obj["hf"][repo_id] = lora_filename
        input_volumes[f"/volume/hf/{repo_id}"] = HuggingfaceInputVolume(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[lora_filename],
        )

    return await _execute_job(
        dataset_url=test_split_url,
        job_obj=job_obj,
        input_volumes=input_volumes,
        docker_image=ch_cst.CH_JOB_VALIDATOR_DOCKER_IMAGE_DIFFUSION,
        is_image=True,
    )


async def run_evaluation_docker_text(
    dataset_url: str,
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType,
    file_format: FileFormat,
    gpu_ids: list[int],
    command: list[str],
    environment: dict[str, str],
) -> dict:
    logger.info("-=+*" * 20)
    logger.info("----- run_evaluation_docker_text -----")
    logger.info(f"{dataset_url=}")
    logger.info(f"{dataset=}")
    logger.info(f"{models=}")
    logger.info(f"{original_model=}")
    logger.info(f"{dataset_type=}")
    logger.info(f"{file_format=}")
    logger.info(f"{gpu_ids=}")
    logger.info("-=+*" * 20)

    env_original_model = environment["ORIGINAL_MODEL"]
    environment["ORIGINAL_MODEL"] = f"/volume/hf/{env_original_model}"

    if ch_cst.HF_REPLACE_NAME_WITH_PATH:
        env_models = environment["MODELS"]
        environment["MODELS"] = ",".join(f"/volume/hf/{model}" for model in env_models.split(","))

    job_obj: dict[str, Any] = dict(
        hf={},
        env=environment,
    )

    input_volumes: dict[str, InputVolume] = {}

    logger.info(f">>>> {original_model}")
    job_obj["hf"][original_model] = None
    input_volumes[f"/volume/hf/{original_model}"] = HuggingfaceInputVolume(
        repo_id=original_model,
        repo_type="model",
    )

    for repo_id in models:
        logger.info(f">>>> {repo_id}")
        job_obj["hf"][repo_id] = None
        input_volumes[f"/volume/hf/{repo_id}"] = HuggingfaceInputVolume(
            repo_id=repo_id,
            repo_type="model",
        )

    if isinstance(dataset_type, GrpoDatasetType):
        if dataset_type.reward_functions is not None:

            def patch_detoxify(match: re.Match) -> str:
                model_type = match.group(1)
                url = detoxify.MODEL_URLS[model_type]
                rel_path = urlparse(url).path.removeprefix("/")

                full_path = f"/volume/detoxify/{rel_path}"
                input_volumes[full_path] = HTTPInputVolume(url=url)

                repo_id = ch_cst.DETOXIFY_MODEL[model_type]
                input_volumes[f"/volume/hf/{repo_id}"] = HuggingfaceInputVolume(
                    repo_id=repo_id,
                    repo_type="model",
                )

                return f"Detoxify('{model_type}', checkpoint='{full_path}')"

            logger.info(">>> REWARD FUNCTIONS:")
            patched = False
            for func in dataset_type.reward_functions:
                logger.info(f"\n{func.reward_func}")
                patched_func = re.sub(
                    r"""Detoxify\(['"](\w+)['"]\)""",
                    patch_detoxify,
                    func.reward_func,
                )
                if func.reward_func != patched_func:
                    func.reward_func = patched_func
                    logger.info(f"PATCHED:\n{func.reward_func}")
                    patched = True
                if "langcheck" in func.reward_func:
                    for metric, model in ch_cst.LANGCHECK_EN_MODEL.items():
                        if f"langcheck.metrics.{metric}" in func.reward_func:
                            model_name = model["model_name"]
                            input_volumes[f"/volume/hf/{model_name}"] = HuggingfaceInputVolume(
                                repo_id=model_name,
                                repo_type="model",
                                revision=model["model_revision"],
                            )
            if patched:
                job_obj["env"]["DATASET_TYPE"] = dataset_type.model_dump_json()

    return await _execute_job(
        dataset_url=dataset_url,
        job_obj=job_obj,
        input_volumes=input_volumes,
        docker_image=ch_cst.CH_JOB_VALIDATOR_DOCKER_IMAGE,
        is_image=False,
        args=command,
    )


async def _execute_job(
    *,
    dataset_url: str,
    job_obj: dict[str, Any],
    input_volumes: dict[str, InputVolume],
    docker_image: str,
    is_image: bool,
    args: list[str] | None = None,
) -> dict:
    dataset_filename = urlparse(dataset_url).path.removeprefix("/")
    job_obj["dataset_filename"] = dataset_filename

    input_volumes[f"/volume/s3/{dataset_filename}"] = HTTPInputVolume(
        url=dataset_url,
    )

    input_volumes["/volume/meta"] = InlineInputVolume.from_file_contents(
        filename="job.json",
        contents=json.dumps(job_obj).encode("utf8"),
    )

    wallet = bittensor_wallet.Wallet(
        name=ch_cst.CH_JOB_SIGNER_WALLET_NAME,
        hotkey=ch_cst.CH_JOB_SIGNER_WALLET_HOTKEY_NAME,
    )

    logger.info(f"Facilitator: {ch_cst.CH_JOB_FACILITATOR_URL}")
    compute_horde_client = ComputeHordeClient(
        hotkey=wallet.hotkey,
        compute_horde_validator_hotkey=ch_cst.CH_JOB_TARGET_VALIDATOR_HOTKEY,
        facilitator_url=ch_cst.CH_JOB_FACILITATOR_URL,
    )

    job_spec = ComputeHordeJobSpec(
        executor_class=ExecutorClass.always_on__llm__a6000,
        job_namespace=ch_cst.CH_JOB_NAMESPACE,
        docker_image=docker_image,
        download_time_limit_sec=900,
        execution_time_limit_sec=1200,
        upload_time_limit_sec=5,
        args=args or [],
        artifacts_dir="/artifacts",
        input_volumes=input_volumes,
    )

    logger.info("%" * 80)
    pprint.pprint(job_obj)
    logger.info("%" * 80)
    pprint.pprint(asdict(job_spec))
    logger.info("%" * 80)

    job = await compute_horde_client.create_job(job_spec)

    while True:
        current_time = datetime.now().strftime("%H:%M:%S")
        logger.info(f"{current_time} {job.uuid} {job.status}")

        if job.status in (
            ComputeHordeJobStatus.COMPLETED,
            ComputeHordeJobStatus.REJECTED,
            ComputeHordeJobStatus.FAILED,
        ):
            break

        job = await compute_horde_client.get_job(job.uuid)
        await asyncio.sleep(3)

    assert job.result is not None
    logger.info(" *** STDOUT " + "*" * 80)
    logger.info("\n" + job.result.stdout.strip())
    logger.info(" *** STDERR " + "*" * 80)
    logger.info("\n" + job.result.stderr.strip())
    logger.info("*" * 80)
    logger.info(f"ARTIFACT COUNT: {len(job.result.artifacts)}")
    for name, value in job.result.artifacts.items():
        logger.info(f"ARTIFACT: {name}")
        if name == "/artifacts/evaluation_results.json":
            # FIXME: circular import
            from validator.evaluation.docker_evaluation import process_evaluation_results

            eval_results_dict = json.loads(value)
            eval_results = process_evaluation_results(eval_results_dict, is_image=is_image)
            logger.info("+" * 80)
            logger.info("\n%s", eval_results_dict)
            logger.info("+" * 80)
            logger.info("\n%s", eval_results.model_dump())
            logger.info("+" * 80)
            return eval_results_dict

    raise RuntimeError("Evaluation results missing from artifacts")
