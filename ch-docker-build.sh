#!/bin/bash

GIT_COMMIT=$(git rev-parse HEAD)
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

docker build \
  -f dockerfiles/validator.dockerfile \
  --build-arg GIT_COMMIT="$GIT_COMMIT" \
  --build-arg BUILD_DATE="$BUILD_DATE" \
  -t weightswandering/tuning_vali:latest \
  -t backenddevelopersltd/ch-sn56-tuning_vali:latest \
  .

docker build \
  -f dockerfiles/validator-diffusion.dockerfile \
  --build-arg GIT_COMMIT="$GIT_COMMIT" \
  --build-arg BUILD_DATE="$BUILD_DATE" \
  -t diagonalge/tuning_validator_diffusion:latest \
  -t backenddevelopersltd/ch-sn56-tuning_validator_diffusion:latest \
  .

# docker push backenddevelopersltd/ch-sn56-tuning_vali:latest
# docker push backenddevelopersltd/ch-sn56-tuning_validator_diffusion:latest
