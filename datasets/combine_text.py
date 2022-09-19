import os

FOLDER_PATH = "/media/allysakate/starplan/Dissertation/Dataset/MOTSChallenge/train/"
DATA_SPLIT = "train"
OUT = os.path.join(FOLDER_PATH, f"{DATA_SPLIT}.txt")
filenames = ["0002", "0005", "0009", "0011"]
with open(OUT, 'w') as outfile:
    for fname in filenames:
        txt_file = os.path.join(FOLDER_PATH, f"{DATA_SPLIT}_{fname}.txt")
        with open(txt_file) as infile:
            for line in infile:
                outfile.write(line)
