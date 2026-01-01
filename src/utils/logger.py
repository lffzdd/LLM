from logging import Logger, StreamHandler, Formatter, DEBUG
def get_logger(name: str) -> Logger:
    logger = Logger(name)
    handler = StreamHandler()
    formatter = Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(DEBUG)
    return logger