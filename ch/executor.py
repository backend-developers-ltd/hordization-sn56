import os
import sys
import json
import pprint
import shutil
import zipfile
import subprocess

import ch.download
from ch.state import JOB_META

from core import constants as cst
from validator.utils.logging import get_logger

logger = get_logger(__name__)


def init() -> None:
    logger.info("+++ Initializing Compute Horde job")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"CMD: {' '.join(sys.argv)}")
    logger.info(f"BUILD_DATE={os.environ['BUILD_DATE']}")
    logger.info(f"GIT_COMMIT={os.environ['GIT_COMMIT']}")

    logger.info("ENVIRONMENT variables:")
    for name, value in sorted(os.environ.items()):
        logger.info(f"  {name}={value}")

    ch.download.patch()

    with open("/volume/meta/job.json", "rb") as f:
        JOB_META.update(json.load(f))

    logger.info("%" * 80)
    pprint.pprint(JOB_META)
    logger.info("%" * 80)

    logger.info(f"PID {os.getpid()} VOLUME BEGIN")
    result = subprocess.run("find /volume | sort", shell=True, capture_output=True, text=True)
    logger.info(f"PID {os.getpid()}\n" + result.stdout.strip())
    logger.info(f"PID {os.getpid()}\n" + result.stderr.strip())
    logger.info(f"PID {os.getpid()} VOLUME END")

    if not os.path.exists("/workspace/input_data"):
        dataset_filename = JOB_META["dataset_filename"]
        dataset_dest_path = JOB_META["env"]["DATASET"]
        dataset_source_path = f"/volume/s3/{dataset_filename}"
        if dataset_filename.endswith(".zip"):
            logger.info(f"+++ Extracting dataset {dataset_filename} to {dataset_dest_path}")
            with zipfile.ZipFile(dataset_source_path, "r") as zip_ref:
                zip_ref.extractall(dataset_dest_path)
        else:
            os.makedirs(os.path.dirname(dataset_dest_path), exist_ok=True)
            logger.info(f"+++ Creating symlink from {dataset_source_path} to {dataset_dest_path}")
            result = subprocess.run(
                ["ln", "-s", dataset_source_path, dataset_dest_path],
                capture_output=True,
                text=True,
            )
            logger.info("\n" + result.stdout.strip())
            logger.info("\n" + result.stderr.strip())
            assert result.returncode == 0

    logger.info(f"PID {os.getpid()} INPUT DATA BEGIN")
    result = subprocess.run("find /workspace/input_data | sort", shell=True, capture_output=True, text=True)
    logger.info(f"PID {os.getpid()}\n" + result.stdout.strip())
    logger.info(f"PID {os.getpid()}\n" + result.stderr.strip())
    logger.info(f"PID {os.getpid()} INPUT DATA END")

    for name, value in JOB_META["env"].items():
        os.environ[name] = value


def process_results() -> None:
    results_file = cst.CONTAINER_EVAL_RESULTS_PATH
    artifacts_file = "/artifacts/" + os.path.basename(results_file)
    logger.info(f"Copying {results_file} to {artifacts_file}")
    shutil.copy(results_file, artifacts_file)

    with open(results_file, "rb") as f:
        results = json.load(f)
    logger.info("+" * 80)
    logger.info("\n%s", results)
    logger.info("+" * 80)
