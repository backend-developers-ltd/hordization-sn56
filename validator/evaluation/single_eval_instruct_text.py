import sys

from ch import constants as ch_cst
import ch.executor

from validator.core.models import EvaluationArgs
from validator.evaluation.eval_instruct_text import evaluate_repo


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python single_eval_instruct_text.py <serialized_evaluation_args>")
        sys.exit(1)

    if ch_cst.CH_VALIDATION_INSTRUCT_TEXT_TASK:
        ch.executor.init()

    evaluation_args = EvaluationArgs.model_validate_json(sys.argv[1])
    evaluate_repo(evaluation_args)
