#!/bin/bash

docker run \
  --rm \
  -it \
  --gpus "device=all" \
  -v ~/tmp/validator/temp_images:/app/avatars/:rw \
  -e SAVE_DIR=/app/avatars/ \
  -e NUM_PROMPTS=10 \
  diagonalge/person_synth:latest
