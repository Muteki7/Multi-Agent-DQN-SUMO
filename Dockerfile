FROM Python:3.14.3

#Always create a working directory so when you run this image like a container you can actually work on it and make changes
WORKDIR /app

COPY . . 
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sumo\
        sumo-tools \
        && rm -rf /var/lib/apt/lists/*

RUN useradd app
USER app

CMD ["python", "evaluation.py"]
