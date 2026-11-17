FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen

COPY ./app /code/app
COPY ./helper_lib /code/helper_lib
COPY ./models /code/models

ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

CMD ["uv", "run", "--frozen", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "80"]
