import math
import os
import sys
from argparse import ArgumentParser
from typing import List
from threading import Thread, Lock
from math import ceil
import re

"""
Fix the ModelNet40 dataset.
It has a known problem of lacking a line break on the header.
"""

n_threads = 8

def process_batch(tidx: int, filenames: List[str], begin: int, end: int, fcount: List[int], fbad: List[int]):

    total = len(filenames)

    for i in range(begin, end):
        # check bound
        if i >= total:
            return

        # read the files and check the header
        """
        expected:
        OFF
        1234 5678 0

        defect:
        OFF1234 5678 0
        """
        fname = filenames[i]
        fullpath = os.path.join(args.path, c, mode, fname)
        # open the file for read
        handle = open(fullpath, "r")
        # read the content
        content = handle.read()
        # close the file
        handle.close()

        # split the content by lines
        content = content.split("\n")
        # check the header for errors
        if content[0].strip() != "OFF":
            fbad[tidx] += 1
            print(f"\nFound file {fullpath} with defect")
            split = re.split(r"([A-Za-z]+)(.*)", content[0], maxsplit=1)
            content[0] = split[1]
            content.insert(1, split[2])
            # create a new string with the content
            content_str = ""
            for l in content:
                content_str += l.strip() + "\n"
            # write the new content to the file
            handle = open(fullpath, "w")
            handle.write(content_str)
            handle.close()
        fcount[tidx] += 1



if __name__ == '__main__':

    # parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    # check if the dataset path exists
    if not os.path.exists(args.path):
        raise FileNotFoundError("The dataset path does not exist.")

    # list the classes
    classes: List[str] = os.listdir(args.path)

    fcount: int = 0
    fbad: int = 0

    # iterate the classes
    for c in classes:
        # iterate the stage (train/test)
        for mode in ["train", "test"]:

            threads: List[Thread] = []
            badcount: List[int] = [0]*n_threads
            totalcount: List[int] = [0]*n_threads


            # get the files list
            f = os.listdir(os.path.join(args.path, c, mode))
            f.sort()

            # divide the samples bby the number of threads
            samples_per_thread = ceil(len(f)/n_threads)

            # start the threads
            begin: int = 0
            for i in range(n_threads):
                t = Thread(target=process_batch, args=(i, f, begin, begin + samples_per_thread, totalcount, badcount))
                t.start()
                threads.append(t)
                begin += samples_per_thread

            # wait for all threads
            for t in threads:
                t.join()

            fcount += sum(totalcount)
            fbad += sum(badcount)

            print(f"Scanned files: {fcount}, Bad: {fbad}", end='\r')

    sys.exit(0)
