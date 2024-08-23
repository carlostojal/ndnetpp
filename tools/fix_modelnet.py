import os
import sys
from argparse import ArgumentParser
from typing import List
import re

"""
Fix the ModelNet40 dataset.
It has a known problem of lacking a line break on the header.
"""

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
            # get the files list
            f = os.listdir(os.path.join(args.path, c, mode))
            f.sort()
            
            for fname in f:
                # read the files and check the header
                """
                expected:
                OFF
                1234 5678 0

                defect:
                OFF1234 5678 0
                """
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
                    fbad += 1
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
                fcount += 1

                print(f"Scanned files: {fcount}, Bad: {fbad}", end='\r')
                
    sys.exit(0)
