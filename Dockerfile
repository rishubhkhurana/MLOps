FROM python:3.9-slim-bullseye

WORKDIR /opt

COPY requirements.txt requirements.txt

RUN pip install --upgrade -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY . .

# ENTRYPOINT [ "./entrypoint.sh" ]

# CMD ["best_model_path='./model.pt'"]