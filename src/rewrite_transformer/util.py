from logging import DEBUG, Formatter, Logger, StreamHandler


def get_logger(name: str) -> Logger:
    logger = Logger(name)
    handler = StreamHandler()
    formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(DEBUG)

    return logger


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    return data