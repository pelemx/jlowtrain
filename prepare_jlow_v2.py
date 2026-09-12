#!/usr/bin/env python3

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC._special_tokens["<|endoftext|>"]

HF_REPO = "taufiqdp/Indo4B-hf"

SOURCES = {
    "indo4b_small": [
        "data/opensubtitles-00000-of-00001.parquet",
        "data/wiki-00000-of-00001.parquet",
        "data/kompas-00000-of-00001.parquet",
        "data/tempo-00000-of-00001.parquet",
    ],
    "indo4b_oscar": [
        "data/oscar_all_uncased-00000-of-00004.parquet",
        "data/oscar_all_uncased-00001-of-00004.parquet",
        "data/oscar_all_uncased-00002-of-00004.parquet",
        "data/oscar_all_uncased-00003-of-00004.parquet",
    ],
    "indo4b_conllu": [
        "data/conllu_all_uncased-00000-of-00002.parquet",
        "data/conllu_all_uncased-00001-of-00002.parquet",
    ],
}


def chat(system, user, assistant):
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n" + assistant + "<|im_end|>\n"
    )


def tool_example(user, tool_name, args, response, answer):
    return (
        chat(
            "Kamu adalah Jlow. Jika butuh tool, keluarkan SATU JSON "
            "tool call dan jangan mengarang hasil tool.",
            user,
            json.dumps(
                {"name": tool_name, "arguments": args},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        + "<|im_start|>tool\n"
        + json.dumps(
            {"name": tool_name, "result": response},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + answer
        + "<|im_end|>\n"
    )


def custom_agent_seeds():
    seeds = [

        # WEATHER
        tool_example(
            "Cuaca Jogja besok gimana?",
            "get_weather",
            {"city": "Yogyakarta"},
            {"temp": 28, "condition": "Partly Cloudy"},
            "Besok Jogja sekitar 28°C dan agak berawan bro.",
        ),

        tool_example(
            "Bro, cek cuaca Jakarta besok dong.",
            "get_weather",
            {"city": "Jakarta"},
            {"temp": 31, "condition": "Cloudy"},
            "Besok Jakarta sekitar 31°C dan berawan.",
        ),

        # WEB SEARCH
        tool_example(
            "Cari info terbaru soal harga BTC.",
            "web_search",
            {"query": "harga BTC terbaru"},
            {"status": "ok", "results": ["BTC sekitar ..."]},
            "Gue nemu hasil terbaru dari web. Cek hasilnya sebelum ambil keputusan.",
        ),

        tool_example(
            "Cari berita terbaru tentang teknologi AI.",
            "web_search",
            {"query": "berita AI terbaru"},
            {"status": "ok", "results": ["AI ..."]},
            "Gue nemu beberapa hasil terbaru soal AI.",
        ),

        # RAG
        tool_example(
            "Cari di knowledge base tentang refund.",
            "rag_search",
            {"query": "refund"},
            {"status": "ok", "results": ["Kebijakan refund ..."]},
            "Di knowledge base ada informasi soal kebijakan refund.",
        ),

        tool_example(
            "Cek aturan membership di knowledge base.",
            "rag_search",
            {"query": "membership"},
            {"status": "ok", "results": ["Aturan membership ..."]},
            "Gue nemu aturan membership di knowledge base.",
        ),

        # SOLANA
        tool_example(
            "Berapa saldo wallet ini?",
            "solana_balance",
            {"address": "EXAMPLE"},
            {"status": "ok", "balance": 1.23},
            "Saldo wallet sekitar 1.23 SOL.",
        ),

        # NO TOOL
        chat(
            "Kamu adalah Jlow, asisten bahasa Indonesia yang santai dan natural.",
            "Bro, lu lagi ngapain?",
            "Lagi santai bro wkwk. Ada yang mau lu bahas?",
        ),

        chat(
            "Kamu adalah Jlow, asisten bahasa Indonesia yang santai.",
            "Wkwk lu bisa bantu gue gak?",
            "Bisa dong bro. Gas, kasih tau aja yang lu butuhin.",
        ),

        chat(
            "Kamu adalah Jlow, asisten bahasa Indonesia yang jelas.",
            "Jelasin internet secara singkat.",
            "Internet adalah jaringan besar yang menghubungkan banyak komputer dan perangkat supaya bisa saling bertukar data.",
        ),

        # TOOL ERROR
        tool_example(
            "Cek cuaca.",
            "get_weather",
            {},
            {"status": "error", "message": "city is required"},
            "Gue butuh nama kotanya dulu bro. Mau cek kota mana?",
        ),

        tool_example(
            "Cari saldo wallet.",
            "solana_balance",
            {"address": "INVALID"},
            {"status": "error", "message": "invalid address"},
            "Alamat wallet-nya nggak valid bro.",
        ),

        # UNKNOWN TOOL
        chat(
            "Kamu adalah Jlow. Gunakan tool hanya jika tool tersebut tersedia.",
            "Pakai tool teleport buat kirim gue ke Bali.",
            "Tool teleport nggak tersedia, jadi gue nggak bisa menjalankan itu.",
        ),

        # JAKSEL
        chat(
            "Kamu adalah Jlow, asisten Indonesia/Jaksel yang natural.",
            "Bro, menurut lu meeting ini worth it gak?",
            "Kalau ada decision penting yang harus dibahas, menurut gue worth it sih. Tapi kalau cuma sync tipis-tipis, bisa jadi agak wasting time.",
        ),

        chat(
            "Kamu adalah Jlow, asisten Indonesia/Jaksel yang santai.",
            "Gue lagi overwhelmed banget sama kerjaan.",
            "Take it easy bro. Coba breakdown jadi task kecil dulu, jangan dipikirin semuanya sekaligus.",
        ),
    ]

    # Oversample seeds.
    return seeds * 100


def get_text(row):
    for key in (
        "text",
        "content",
        "sentence",
        "document",
        "raw",
        "normalized",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key, value in row.items():
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def load_streaming_rows(path, max_rows):
    uri = f"hf://datasets/{HF_REPO}/{path}"

    ds = load_dataset(
        "parquet",
        data_files=uri,
        split="train",
        streaming=True,
    )

    out = []

    for row in ds:
        text = get_text(row)

        if text:
            out.append(text)

        if max_rows and len(out) >= max_rows:
            break

    return out


def load_chat(max_rows):
    ds = load_dataset(
        "LorthGyu/indonesian-chat",
        split="train",
    )

    out = []

    for row in ds:
        conv = row.get("conversations", [])
        messages = []

        if isinstance(conv, list):
            for message in conv:
                if (
                    isinstance(message, dict)
                    and message.get("role") in ("user", "assistant")
                    and isinstance(message.get("content"), str)
                ):
                    messages.append(message)

        if not messages:
            continue

        body = (
            "<|im_start|>system\n"
            "Kamu adalah Jlow, asisten bahasa Indonesia yang natural "
            "dan mengikuti konteks percakapan."
            "<|im_end|>\n"
        )

        for message in messages:
            body += (
                f"<|im_start|>{message['role']}\n"
                f"{message['content']}<|im_end|>\n"
            )

        out.append(body)

        if max_rows and len(out) >= max_rows:
            break

    return out


def load_robo(max_rows):
    ds = load_dataset(
        "Caplin43/IndoRobo-Instruction-Dataset-v1",
        split="train",
    )

    out = []

    for row in ds:
        instruction = str(row.get("instruction", "")).strip()
        action = str(row.get("action", "")).strip()

        if instruction and action:
            out.append(
                chat(
                    "Kamu adalah Jlow. Ubah instruksi Indonesia "
                    "menjadi satu JSON action yang valid.",
                    instruction,
                    action,
                )
            )

        if max_rows and len(out) >= max_rows:
            break

    return out


def weighted_sample(pools, total, seed=42):
    weights = {
        "indo4b": 40,
        "chat": 20,
        "instruction": 5,
        "agent": 35,
    }

    rng = random.Random(seed)

    names = [name for name in weights if pools.get(name)]

    if not names:
        raise RuntimeError("Tidak ada dataset pool yang tersedia.")

    probs = [weights[name] for name in names]

    output = []

    for _ in range(total):
        name = rng.choices(
            names,
            weights=probs,
            k=1,
        )[0]

        output.append(
            rng.choice(pools[name])
        )

    return output, weights


def write_tokens(texts, path, label):
    """
    Tokenize dan langsung tulis chunk ke disk.
    Tidak menampung seluruh dataset token di RAM.
    """

    chunk_size = 500
    total_tokens = 0

    path = Path(path)

    if path.exists():
        path.unlink()

    with open(path, "wb") as f:

        total = len(texts)

        for start in range(0, total, chunk_size):

            chunk = texts[start:start + chunk_size]

            ids = []

            for text in chunk:
                ids.extend(
                    ENC.encode_ordinary(text)
                )
                ids.append(EOT)

            arr = np.asarray(
                ids,
                dtype=np.uint16,
            )

            arr.tofile(f)

            total_tokens += len(arr)

            done = min(
                start + len(chunk),
                total,
            )

            if done % 5000 < chunk_size or done == total:
                print(
                    f"{label}: "
                    f"{done:,}/{total:,} samples | "
                    f"{total_tokens:,} tokens",
                    flush=True,
                )

    return total_tokens


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        default="data/jlow_v2",
    )

    parser.add_argument(
        "--indo4b-small-per-file",
        type=int,
        default=25000,
    )

    parser.add_argument(
        "--indo4b-oscar-per-file",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--indo4b-conllu-per-file",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--chat-max",
        type=int,
        default=877,
    )

    parser.add_argument(
        "--robo-max",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200000,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    pools = {
        "indo4b": [],
        "chat": [],
        "instruction": [],
        "agent": [],
    }

    # ========================================================
    # INDO4B SMALL
    # ========================================================

    print("== Indo4B small sources ==")

    for path in SOURCES["indo4b_small"]:

        rows = load_streaming_rows(
            path,
            args.indo4b_small_per_file,
        )

        pools["indo4b"].extend(rows)

        print(
            path,
            len(rows),
            flush=True,
        )

    # ========================================================
    # OSCAR
    # ========================================================

    print("== Indo4B OSCAR ==")

    for path in SOURCES["indo4b_oscar"]:

        rows = load_streaming_rows(
            path,
            args.indo4b_oscar_per_file,
        )

        pools["indo4b"].extend(rows)

        print(
            path,
            len(rows),
            flush=True,
        )

    # ========================================================
    # CONLLU
    # ========================================================

    print("== Indo4B CONLLU ==")

    for path in SOURCES["indo4b_conllu"]:

        rows = load_streaming_rows(
            path,
            args.indo4b_conllu_per_file,
        )

        pools["indo4b"].extend(rows)

        print(
            path,
            len(rows),
            flush=True,
        )

    # ========================================================
    # CHAT
    # ========================================================

    print("== Indonesian chat ==")

    pools["chat"] = load_chat(
        args.chat_max
    )

    print(
        len(pools["chat"]),
        flush=True,
    )

    # ========================================================
    # INDROROBO
    # ========================================================

    print("== IndoRobo ==")

    pools["instruction"] = load_robo(
        args.robo_max
    )

    print(
        len(pools["instruction"]),
        flush=True,
    )

    # ========================================================
    # AGENT
    # ========================================================

    print("== Custom agent/style seeds ==")

    pools["agent"] = custom_agent_seeds()

    print(
        len(pools["agent"]),
        flush=True,
    )

    # ========================================================
    # POOL REPORT
    # ========================================================

    print()
    print("== Dataset pools ==")

    for name, pool in pools.items():
        print(
            f"{name:15s}: {len(pool):,}",
            flush=True,
        )

    # ========================================================
    # WEIGHTED SAMPLING
    # ========================================================

    print()
    print("== Weighted sampling ==")

    texts, weights = weighted_sample(
        pools,
        args.samples,
        SEED,
    )

    random.Random(SEED).shuffle(texts)

    n_val = max(
        1,
        int(len(texts) * args.val_ratio),
    )

    val_texts = texts[:n_val]
    train_texts = texts[n_val:]

    print(
        f"train samples : {len(train_texts):,}"
    )

    print(
        f"val samples   : {len(val_texts):,}"
    )

    print(
        f"weights       : {weights}"
    )

    # ========================================================
    # TOKENIZE + WRITE
    # ========================================================

    print()
    print("== Tokenizing train ==")

    train_tokens = write_tokens(
        train_texts,
        out / "train.bin",
        "TRAIN",
    )

    print()
    print("== Tokenizing val ==")

    val_tokens = write_tokens(
        val_texts,
        out / "val.bin",
        "VAL",
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {
        "tokenizer": "gpt2",
        "vocab_size": 50257,
        "seed": SEED,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": train_tokens + val_tokens,
        "dtype": "uint16",
        "format": "Jlow chat serialization + raw Indonesian text",
        "datasets": {
            name: len(pool)
            for name, pool in pools.items()
        },
        "weights": weights,
        "skipped": [
            {
                "dataset": "jakartaresearch/indoqa",
                "reason": "legacy dataset script unsupported; intentionally skipped",
            }
        ],
    }

    manifest_path = out / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("DATASET BUILD FINISHED")
    print("=" * 70)

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("OUTPUT:")

    print(out / "train.bin")
    print(out / "val.bin")
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
