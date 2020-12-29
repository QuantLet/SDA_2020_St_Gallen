# %% Setup
import os
import pandas as pd
import numpy as np
import time
import datetime

import cv2
import pytesseract
from pdf2image import convert_from_path


# %% Main
def main():
    start = time.time()

    # i) specify cols for final df
    columns = ['date', 'text', 'pagenr', 'filename']
    data = []

    pdf_dir = "../SDA_2020_St_Gallen/SDA_2020_St_Gallen_SentimentAnalysis_SNB_Statements/data/lagebeurteilungenSNB/"

    # ii) loop through pdf files
    for i, fname in enumerate(os.listdir(pdf_dir)):
        print("Extracting text from '{}' ({}/{})".format(fname, i+1, len(os.listdir(pdf_dir))))
        path = pdf_dir + fname
        imgs = convert_from_path(path)

        # iii) loop page-images and apply OCR
        for i, img in enumerate(imgs):
            # scale images to increase OCR accuracy
            height, width = img.size
            img = np.array(img)
            img = cv2.resize(img, (3 * width, 3 * height), interpolation=cv2.INTER_LINEAR)
            img = cv2.bitwise_not(img)

            # tesseract needs to be installed + german training data has to be downloaded and put into
            # C:\ProgramData\Anaconda3\envs\*env_name*\Library\bin\tessdata
            # can be downloaded from https://github.com/tesseract-ocr/tessdata/blob/master/deu.traineddata
            text = pytesseract.image_to_string(img, lang='deu')
            date_str = fname[4:12]  # cut out 8 digit date
            date = datetime.datetime.strptime(date_str, '%Y%m%d')
            data.append([date, text, i, fname])

    # save final df
    df = pd.DataFrame(data, columns=columns)
    df.to_excel('data/articles_raw.xlsx', engine='xlsxwriter')

    print("Saved text in DataFrame. Elapsed time: {}".format(time.strftime("%Mm %Ss", time.gmtime(time.time()-start))))


# %% Run file
if __name__ == '__main__':
    main()
