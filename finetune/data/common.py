import json
import logging
import multiprocessing
import os

import torch
import torch.utils.data as torch_data
from transformers.data.processors.squad import SquadExample, squad_convert_examples_to_features
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# from data import cls as cls_data
from data import qa as qa_data
# from datasets import load_dataset
import pandas as pd

DATA_PROCESSOR = {"cls": "cls_data", "qa": qa_data}

def _create_examples(file_path: str, is_training: bool = True, data_split_ratio: int = 1):
    input_data = pd.read_csv(file_path)
    start_pos, end_pos = 0, len(input_data)
    if data_split_ratio != 1:
        split_ratio = (data_split_ratio % 100) / 100
        chunk = int(len(input_data) * split_ratio)
        start_pos = int(chunk * (data_split_ratio // 100))
        end_pos = int(start_pos + chunk)
        
    examples = []
    for i, entry in tqdm(enumerate(input_data[start_pos:end_pos].itertuples())):
        example = SquadExample(
            qas_id="aihub-mrc-v1_train_"+format(i, '06'),
            question_text=entry.question,
            context_text=entry.text,
            answer_text=entry.answer,
            start_position_character=entry.answer_start,
            title=entry.title,
            is_impossible=False,
        )
        examples.append(example)

    return examples

def get_data(config, tokenizer, is_train=True, overwrite=False):
    """Initializes data. Uses to construct features and dataset."""
    if is_train:
        data_file = config.train_file
    else:
        data_file = config.predict_file

    data_path = config.data_dir
    if data_file is not None:
        data_path = os.path.join(data_path, data_file)
    else:
        data_path += "/"

    examples = _create_examples(data_path, is_training=is_train, data_split_ratio=1)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        doc_stride=config.doc_stride,
        max_query_length=config.max_query_length,
        is_training=is_train,
        return_dataset="pt",
        threads=10,
    )

    logger.info(f"Prepare {'Train' if is_train else 'Test'} dataset (Count: {len(dataset)}) ")
    data_processor = DATA_PROCESSOR.get(config.task, None)

    return torch_data.DataLoader(
        dataset,
        batch_size=config.train_batch_size if is_train else config.eval_batch_size,
        collate_fn=(data_processor.collate_fn),
        drop_last=False
    )


def _get_data(config, tokenizer, is_train=True, overwrite=False):
    if is_train:
        data_file = config.train_file
    else:
        data_file = config.predict_file

    data_path = config.data_dir
    if data_file is not None:
        data_path = os.path.join(data_path, data_file)
    else:
        data_path += "/"

    data_processor = DATA_PROCESSOR.get(config.task, None)
    if data_processor is None:
        raise Exception(f"Invalid data task {config.task}!")

    processor = data_processor.process_map.get(config.dataset, None)
    if processor is None:
        raise Exception(f"Invalid task dataset {config.dataset}!")

    comps = [
        data_path,
        config.dataset,
        config.model_name_or_path.replace("/", "_"),
        config.max_seq_length,
        "train" if is_train else "dev",
        "dataset.txt",
    ]
    dataset_file = "_".join([str(comp) for comp in comps])

    if not os.path.exists(dataset_file) or overwrite:
        with open(dataset_file, "w", encoding="utf-8") as writer_file:
            if data_file is None or not os.path.isdir(data_path):
                data = processor(config, data_path, is_train)
                cnt = write_samples(
                    config, tokenizer, is_train, data_processor, writer_file, data, workers=config.threads
                )
            else:
                cnt = 0
                for filename in sorted([f for f in os.listdir(data_path) if f.endswith(".json")]):
                    data = processor(config, os.path.join(data_path, filename), is_train)
                    cnt += write_samples(
                        config, tokenizer, is_train, data_processor, writer_file, data, workers=config.threads
                    )
            logger.info(f"{cnt} features processed from {data_path}")

    # dataset = load_dataset("text", data_files=dataset_file)["train"]
    # dataset = dataset.map(lambda x: json.loads(x["text"]), batched=False)
    dataset = pd.read_csv(data_path)
    dataset = list(dataset["text"])

    if not is_train:
        # for valid datasets, we pad datasets so that no sample will be skiped in multi-device settings
        dataset = IterableDatasetPad(
            dataset=dataset,
            batch_size=config.train_batch_size if is_train else config.eval_batch_size,
            num_devices=config.world_size,
            seed=config.seed,
        )

    dataloader = torch_data.DataLoader(
        dataset,
        sampler=torch_data.RandomSampler(dataset) if is_train else None,
        drop_last=False,
        batch_size=config.train_batch_size if is_train else config.eval_batch_size,
        collate_fn=(data_processor.collate_fn),
    )

    return dataloader


config = None
tokenizer = None
is_train = None
writer = None


def init_sample_writer(_config, _tokenizer, _is_train, _writer):
    global config
    global tokenizer
    global is_train
    global writer
    config = _config
    tokenizer = _tokenizer
    is_train = _is_train
    writer = _writer


def sample_writer(data):
    global config
    global tokenizer
    global is_train
    global writer
    return writer(data, config, tokenizer, is_train)


def write_samples(config, tokenizer, is_train, processor, writer_file, data, workers=4):
    write_cnt = 0
    with multiprocessing.Pool(
        processes=workers,
        initializer=init_sample_writer,
        initargs=(config, tokenizer, is_train, processor.sample_writer),
    ) as pool:
        for write_data in tqdm(
            pool.imap(sample_writer, data), total=len(data), dynamic_ncols=True, desc="writing samples..."
        ):
            if isinstance(write_data, list):
                for datum in write_data:
                    writer_file.write(json.dumps(datum) + "\n")
                write_cnt += len(write_data)
            else:
                writer_file.write(json.dumps(write_data) + "\n")
                write_cnt += 1
    return write_cnt


class IterableDatasetPad(torch_data.IterableDataset):
    def __init__(
        self,
        dataset: torch_data.IterableDataset,
        batch_size: int = 1,
        num_devices: int = 1,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.num_examples = 0

        chunk_size = self.batch_size * num_devices
        length = len(dataset)
        self.length = length + (chunk_size - length % chunk_size)

    def __len__(self):
        return self.length

    def __iter__(self):
        self.num_examples = 0
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            self.dataset.generator.manual_seed(self.seed + self.epoch)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            self.num_examples += 1
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == self.batch_size:
                for batch in current_batch:
                    yield batch
                    if first_batch is None:
                        first_batch = batch.copy()
                current_batch = []

        # pad the last batch with elements from the beginning.
        while self.num_examples < self.length:
            add_num = self.batch_size - len(current_batch)
            self.num_examples += add_num
            current_batch += [first_batch] * add_num
            for batch in current_batch:
                yield batch
            current_batch = []
