FROM ultralytics/ultralytics:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y opencv-python-headless && \
    pip install opencv-contrib-python

ARG DEV=False

COPY pyproject.toml poetry.lock ./
COPY src ./src/

RUN pip install poetry==2.3.3 && \
    poetry config virtualenvs.create false

RUN if [ "$DEV" = "true" ]; then \
     poetry install --with dev --no-root; \
    else \
     poetry install --without dev --no-root; \
    fi && \
    rm -rf /root/.cache/pypoetry

CMD ["/bin/bash"]
