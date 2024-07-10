
FROM thehale/python-poetry:1.8.2-py3.10-slim

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y openssh-client ntp
RUN poetry install

EXPOSE 9090

# Run the application
CMD ["poetry", "run", "python", "-m", "robbieg2.server"]
