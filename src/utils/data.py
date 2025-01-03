import os
import subprocess
from pathlib import Path

from datasets import Audio, Dataset, load_dataset, Value
from huggingface_hub import hf_hub_download


def prepare_librispeech(cache_dir) -> tuple[Dataset, Dataset]:
    raw = load_dataset("openslr/librispeech_asr", "clean", cache_dir=cache_dir)
    processed = raw.remove_columns(["chapter_id"])
    processed = processed.cast_column("speaker_id", Value("string"))
    return processed["train.100"], processed["validation"]


def prepare_tedlium(cache_dir) -> tuple[Dataset, Dataset]:
    raw = load_dataset("LIUM/tedlium", "release1", cache_dir=cache_dir)
    processed = raw.remove_columns(["gender"])
    return processed["train"], processed["validation"]


def prepare_parler_tts(cache_dir) -> tuple[Dataset, Dataset]:
    raw_mls = load_dataset("parler-tts/mls_eng", cache_dir=cache_dir)
    processed_mls = raw_mls.remove_columns(
        ["begin_time", "end_time", "speaker_id", "book_id", "audio_duration"]
    )
    processed_mls = processed_mls.rename_column("transcript", "text")

    return processed_mls["train"], processed_mls["dev"]


def prepare_synthetic(cache_dir) -> tuple[Dataset, Dataset]:
    raw = load_dataset("homebrewltd/instruction-speech-encodec-v1", cache_dir=cache_dir)
    processed = raw.remove_columns(["prompt", "length"])
    processed = processed.rename_column("answer", "text")
    splits = processed["train"].train_test_split(test_size=0.1)

    return splits["train"], splits["test"]


def prepare_parler_tts_with_description(cache_dir) -> tuple[Dataset, Dataset]:
    audio = load_dataset("parler-tts/libritts_r_filtered", "clean", cache_dir=cache_dir)
    train_audio, val_audio = audio["train.clean.100"], audio["dev.clean"]

    columns = ["id", "text", "path", "text_description"]
    raw = load_dataset(
        "parler-tts/libritts-r-filtered-speaker-descriptions",
        "clean",
        cache_dir=cache_dir,
    )
    processed = raw.remove_columns(
        list(set(raw.column_names["dev.clean"]) - set(columns))
    )
    train_text, val_text = processed["train.clean.100"], processed["dev.clean"]

    assert train_audio["id"] == train_text["id"] and val_audio["id"] == val_text["id"]

    audio_features_train = train_audio["audio"]
    audio_features_val = val_audio["audio"]

    train_text = train_text.map(
        lambda x, i: {"audio": audio_features_train[i]},
        with_indices=True,
        cache_file_name="cache/merge_train",
    )
    val_text = val_text.map(
        lambda x, i: {"audio": audio_features_val[i]},
        with_indices=True,
        cache_file_name="cache/merge_val",
    )
    return train_text, val_text


def prepare_homebrewltd(cache_dir) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(
        "homebrewltd/instruction-speech-encodec-v1", "default", cache_dir=cache_dir
    )["train"]

    dataset = dataset.rename_column("answer", "text")
    splits = dataset.train_test_split(test_size=0.1)

    return splits["train"], splits["test"]


def prepare_emilia(cache_dir) -> tuple[Dataset, Dataset]:
    repo_id = "amphion/Emilia-Dataset"
    file_list = [f"EN/EN-B{str(i).zfill(6)}.tar" for i in range(200)]

    dataset = load_dataset(
        repo_id, data_files={"en": file_list}, split="en", cache_dir=cache_dir
    )
    shuffled = dataset.shuffle(seed=42)
    subset = shuffled.select(range(100_000))
    subset = subset.map(lambda row: {"text": row["json"]["text"]})
    subset = subset.rename_columns({"__key__": "index", "mp3": "audio"})
    splits = subset.train_test_split(test_size=0.1, seed=42)
    return splits["train"], splits["test"]


def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir="/tmp/musiccaps",
    num_attempts=5,
    url_base="https://www.youtube.com/watch?v=",
):
    status = False

    command = f"""
        yt-dlp --quiet --force-keyframes-at-cuts --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" "{url_base}{video_identifier}"
    """.strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, "Downloaded"


def prepare_musiccaps(cache_dir: str) -> tuple[Dataset, Dataset]:
    ds = load_dataset("google/MusicCaps", split="train")
    sampling_rate = 44100
    limit = None
    num_proc, writer_batch_size = 16, 1000

    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = "../music_data"
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, log = download_clip(
                example["ytid"],
                outfile_path,
                example["start_s"],
                example["end_s"],
            )

        example["audio"] = outfile_path
        example["download_status"] = status
        return example

    ds = ds.rename_column("caption", "text")
    ds = ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False,
    )
    ds = ds.filter(lambda x: x["download_status"]).cast_column(
        "audio", Audio(sampling_rate=sampling_rate)
    )

    splits = ds.train_test_split(test_size=0.1, seed=42)
    return splits["train"], splits["test"]


DATASET_2_LOAD_FUNCTION = {
    "emilia": prepare_emilia,
    "homebrewltd": prepare_homebrewltd,
    "librispeech": prepare_librispeech,
    "musiccaps": prepare_musiccaps,
    "parler-tts": prepare_parler_tts,
    "parler_tts_with_description": prepare_parler_tts_with_description,
    "synthetic": prepare_synthetic,
    "tedlium": prepare_tedlium,
}
