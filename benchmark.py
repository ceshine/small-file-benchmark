"""Simple Benchmark of Reading Small Files From Disk

Usage:
  benchmark.py (-h | --help)
  benchmark.py init COUNT
  benchmark.py (create|test) (flat|two_level|four_level|memmap) [--size=<size>]

Arguments:
  COUNT       The number of files to be created. Supports scientific notation (e.g. 3e5).

Options:
  -h --help     Show this screen.
  --size=<size> The size of random generated arrays [default:256].

"""
from pathlib import Path
import uuid
import time

import numpy as np
from tqdm import tqdm
from docopt import docopt

Path("cache").mkdir(exist_ok=True)


def create_filelist(n):
    np.save(
        "cache/filelist.npy",
        np.array([uuid.uuid4().hex for _ in range(n)]))
    # Test
    files = np.load("cache/filelist.npy")
    assert files.shape[0] == n


def create_files_flat(size):
    Path("cache/flat/").mkdir(exist_ok=True)
    files = np.load("cache/filelist.npy")
    np.random.seed(515)
    for name in tqdm(files):
        np.save(
            f"cache/flat/{name}.npy",
            np.random.random(
                int(10 + np.random.random() * 5 * size
                    )).astype(np.float32)
        )


def create_files_two_level(size):
    Path("cache/2level/").mkdir(exist_ok=True)
    files = np.load("cache/filelist.npy")
    for i in range(16**3):
        Path("cache/2level/%03x" % i).mkdir(exist_ok=True)
    np.random.seed(515)
    for name in tqdm(files):
        np.save(
            f"cache/2level/{name[-3:]}/{name}.npy",
            np.random.random(
                int(10 + np.random.random() * 5 * size
                    )).astype(np.float32)
        )


def create_files_four_level(size):
    Path("cache/4level/").mkdir(exist_ok=True)
    files = np.load("cache/filelist.npy")
    for i in range(16):
        for j in range(16):
            for k in range(16):
                Path(f"cache/4level/{i:x}/{j:x}/{k:x}/").mkdir(
                    exist_ok=True, parents=True)
    np.random.seed(515)
    for name in tqdm(files):
        np.save(
            f"cache/4level/{name[-1]}/{name[-2]}/{name[-3]}/{name}.npy",
            np.random.random(
                int(10 + np.random.random() * 5 * size
                    )).astype(np.float32)
        )


def create_files_memmap(size):
    Path("cache/memmap/").mkdir(exist_ok=True)
    files = np.load("cache/filelist.npy")
    arr = np.memmap(
        "cache/memmap/arr.npy", mode="w+", order="C",
        dtype="float32", shape=(files.shape[0], size))
    np.random.seed(515)
    for i in tqdm(range(files.shape[0])):
        arr[i] = np.random.random(size).astype(np.float32)
    arr.flush()


def test_flat(size):
    print("Testing flat structure...")
    files = np.load("cache/filelist.npy")

    # # Check if the size match
    # tmp = np.load(f"cache/flat/{files[0]}.npy")
    # assert tmp.shape[0] == size

    np.random.seed(515)
    np.random.shuffle(files)
    means = []
    start_time = time.time()
    for name in tqdm(files):
        means.append(np.mean(
            np.load(f"cache/flat/{name}.npy")))
    print(np.max(means), np.mean(means), np.min(means))
    print(f"Took {(time.time() - start_time) / 60:.2f} Minutes")


def test_two_level(size):
    print("Testing two-level structure...")
    files = np.load("cache/filelist.npy")

    # # Check if the size match
    # tmp = np.load(f"cache/2level/{files[0][-3:]}/{files[0]}.npy")
    # assert tmp.shape[0] == size

    np.random.seed(515)
    np.random.shuffle(files)
    means = []
    start_time = time.time()
    for name in tqdm(files):
        means.append(np.mean(
            np.load(f"cache/2level/{name[-3:]}/{name}.npy")))
    print(np.max(means), np.mean(means), np.min(means))
    print(f"Took {(time.time() - start_time) / 60:.2f} Minutes")


def test_four_level(size):
    print("Testing four-level structure...")
    files = np.load("cache/filelist.npy")

    # # Check if the size match
    # tmp = np.load(
    #     f"cache/4level/{files[0][-1]}/{files[0][-2]}"
    #     f"/{files[0][-3]}/{files[0]}.npy")
    # assert tmp.shape[0] == size

    np.random.seed(515)
    np.random.shuffle(files)
    means = []
    start_time = time.time()
    for name in tqdm(files):
        means.append(np.mean(
            np.load(
                f"cache/4level/{name[-1]}/"
                f"{name[-2]}/{name[-3]}/{name}.npy"
            )))
    print(np.max(means), np.mean(means), np.min(means))
    print(f"Took {(time.time() - start_time) / 60:.2f} Minutes")


def test_memmap(size):
    files = np.load("cache/filelist.npy")
    means = []
    start_time = time.time()
    arr = np.memmap(
        "cache/memmap/arr.npy", mode="r", order="C", dtype="float32",
        shape=(files.shape[0], size))
    idx = np.arange(files.shape[0])
    np.random.seed(515)
    np.random.shuffle(idx)
    for i in tqdm(idx):
        means.append(np.mean(arr[i]))
    print(np.max(means), np.mean(means), np.min(means))
    print(f"Took {(time.time() - start_time) / 60:.2f} Minutes")


if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    if arguments["init"]:
        create_filelist(int(eval(arguments["COUNT"])))
    elif arguments["create"]:
        if arguments["flat"]:
            create_files_flat(int(arguments["--size"]))
        elif arguments["two_level"]:
            create_files_two_level(int(arguments["--size"]))
        elif arguments["four_level"]:
            create_files_four_level(int(arguments["--size"]))
        else:
            create_files_memmap(int(arguments["--size"]))
    elif arguments["test"]:
        if arguments["flat"]:
            test_flat(int(arguments["--size"]))
        elif arguments["two_level"]:
            test_two_level(int(arguments["--size"]))
        elif arguments["four_level"]:
            test_four_level(int(arguments["--size"]))
        else:
            test_memmap(int(arguments["--size"]))
