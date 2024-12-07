from datasets import Dataset, load_dataset, Value


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


def prepare_audio_captions(cache_dir) -> tuple[Dataset, Dataset]:
    train = AudioCaps(
        root=cache_dir,
        subset="train",
        download=False,
        audio_format="wav",
        download_audio=False,  # this will only download labels and metadata files
    )
    val = AudioCaps(
        root=cache_dir,
        subset="val",
        download=False,
        audio_format="wav",
        download_audio=False,  # this will only download labels and metadata files
    )

    return train, val


DATASET_2_LOAD_FUNCTION = {
    "audiocaps": prepare_audio_captions,
    "homebrewltd": prepare_homebrewltd,
    "librispeech": prepare_librispeech,
    "parler-tts": prepare_parler_tts,
    "parler_tts_with_description": prepare_parler_tts_with_description,
    "synthetic": prepare_synthetic,
    "tedlium": prepare_tedlium,
}
