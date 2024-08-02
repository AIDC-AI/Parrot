import dataclasses
import json
import logging
import logging.handlers
import os
import sys

import requests

from parrot.utils.constants import LOGDIR, BEGIN_LINE, END_LINE
from importlib import import_module
from polyleven import levenshtein
from typing import Tuple

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def rank0_print(*args):
    if int(os.getenv("LOCAL_PROCESS_RANK", 0)) == 0:
        print(*args)

def import_class_from_string(full_class_string):
    # Split the path to get separate module and class names
    module_path, _, class_name = full_class_string.rpartition('.')

    # Import the module using the module path
    module = import_module(module_path)

    # Get the class from the imported module
    cls = getattr(module, class_name)
    return cls


def name2data(name):
    name2path = {
        'llava-pretrain-558k': '/data/mllm_datasets/meta_files/llava-pretrain-558k.json',
        'laion-12k': '/data/mllm_datasets/meta_files/laion_12k.json',
        'cc12m-645k': "/data/mllm_datasets/meta_files/cc12m_645k.json",
        'llava-finetune-665k': '/data/mllm_datasets/meta_files/llava-finetune-665k.json',
        'sharegpt4v-sft-zh': '/data/mllm_datasets/meta_files/sharegpt4v_sft_zh_71k.json',
        'sharegpt4v-sft-pt': '/data/mllm_datasets/meta_files/sharegpt4v_sft_pt_14k.json',
        'sharegpt4v-sft-ar': '/data/mllm_datasets/meta_files/sharegpt4v_sft_ar_12k.json',
        'sharegpt4v-sft-tr': '/data/mllm_datasets/meta_files/sharegpt4v_sft_tr_17k.json',
        'sharegpt4v-sft-ru': '/data/mllm_datasets/meta_files/sharegpt4v_sft_ru_14k.json'
    }
    return json.load(open(name2path[name], "r"))


def print_args(args):
    print(f'\n{BEGIN_LINE}')
    print(f'Parsed Args:')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print(f'{END_LINE}\n')


def decode_json_line_by_line(content):
    def _decode_key(key: str):
        key = key.strip()
        idx = len(key) - 1
        while idx >= 0 and key[idx] in ['"', "'", '{', '}', '[', ']']:
            idx -= 1
        if idx < 0:
            return None
        key = key[:idx + 1]
        idx = len(key) - 1
        while idx >= 0:
            if key[idx] == '"' and (idx == 0 or key[idx - 1] != '\\'):
                break
            else:
                idx -= 1
        key = key[idx + 1:].strip()
        if len(key) > 0:
            return key
        else:
            return None

    def _decode_value(value: str):
        value = value.strip()
        idx = 0
        while idx < len(value) and value[idx] in ['"', "'", '{', '}', '[', ']']:
            idx += 1
        if idx == len(value):
            return None
        value = value[idx:]
        idx = 0
        while idx < len(value):
            if value[idx] == '"' and (idx == len(value) - 1 or value[idx - 1] != '\\'):
                break
            else:
                idx += 1
        value = value[:idx].strip().rstrip(',').strip()
        if len(value) > 0:
            return value
        else:
            return None

    if not isinstance(content, str) or len(content.strip()) == 0:
        return {}
    content = content.strip()
    lines = [line.strip() for line in content.split('\n')]
    parsed_dict = {}
    for line in lines:
        colon_idx = line.find(':')
        if colon_idx <= 0 or colon_idx == len(line) - 1:
            continue
        key = _decode_key(line[:colon_idx])
        value = _decode_value(line[colon_idx + 1:])
        if key and value:
            parsed_dict[key] = value
    return parsed_dict


def levenshtein_search(generation, candidates) -> Tuple[str, float]:
    generation = generation.strip().lower()

    for candidate in candidates:
        if candidate.strip().lower() == generation:
            return candidate, 1.0

    min_dist = None
    best_candidate = None

    for candidate in candidates:
        dist = levenshtein(generation, candidate.strip().lower())
        if min_dist is None or dist < min_dist:
            min_dist = dist
            best_candidate = candidate

    upper_bound = max(len(generation), len(best_candidate.strip().lower()))
    confidence = 1.0 - min_dist * 1.0 / upper_bound

    return best_candidate, confidence
