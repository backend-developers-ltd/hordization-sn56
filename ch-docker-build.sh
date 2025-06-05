#!/bin/bash

docker build \
  -f dockerfiles/validator.dockerfile \
  -t weightswandering/tuning_vali:latest \
  -t backenddevelopersltd/ch-sn56-tuning_vali:latest \
  .

docker build \
  -f dockerfiles/validator-diffusion.dockerfile \
  -t diagonalge/tuning_validator_diffusion:latest \
  -t backenddevelopersltd/ch-sn56-tuning_validator_diffusion:latest \
  .

# docker push backenddevelopersltd/ch-sn56-tuning_vali:latest
# docker push backenddevelopersltd/ch-sn56-tuning_validator_diffusion:latest
