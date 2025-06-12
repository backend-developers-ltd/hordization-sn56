FROM winglian/axolotl:main-20250429

WORKDIR /app

COPY validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV JOB_ID=""
ENV DATASET=""
ENV MODELS=""
ENV ORIGINAL_MODEL=""
ENV DATASET_TYPE=""
ENV FILE_FORMAT=""

RUN mkdir /aplp

ARG GIT_COMMIT=""
ENV GIT_COMMIT="${GIT_COMMIT}"
ARG BUILD_DATE=""
ENV BUILD_DATE="${BUILD_DATE}"
