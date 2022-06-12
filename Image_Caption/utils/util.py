import sys
from loguru import logger
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def get_logger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


def ptb_tokenize(key_to_captions):
    captions_for_image = {}
    for key, caps in key_to_captions.items():
        captions_for_image[key] = []
        for idx, cap in enumerate(caps):
            captions_for_image[key].append({
                # "image_id": key
                # "id": idx,
                "caption": cap
            })
    tokenizer = PTBTokenizer()
    key_to_captions = tokenizer.tokenize(captions_for_image)
    return key_to_captions