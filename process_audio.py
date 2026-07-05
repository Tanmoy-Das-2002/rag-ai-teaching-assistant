# Renames mp3 files: "Title ｜ ... - Day #1.mp3" -> "1_Title.mp3"
import os

folder = "audios"  # change to your folder name
files = os.listdir(folder)

for file in files:
    tutorial_number = file.split(" #")[1].split(".mp3")[0]
    file_name = file.split(" ｜ ")[0]
    print(tutorial_number, file_name)
    os.rename(f"{folder}/{file}", f"{folder}/{tutorial_number}_{file_name}.mp3")