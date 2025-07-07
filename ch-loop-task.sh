#!/bin/bash

for task_id in "$@"; do

  for i in {1..5}; do
    .venv/bin/python -m utils.run_evaluation --task_id "$task_id"
    sleep 3
  done

  for i in {1..200}; do
    .venv/bin/python -m utils.run_evaluation --task_id "$task_id" --random-seed
    sleep 3
  done

done
