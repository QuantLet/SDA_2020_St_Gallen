# %% Setup
import os
import pandas as pd
import numpy as np
import time
import datetime

from bs4 import BeautifulSoup
import urllib.request
import requests

from pdf2image import convert_from_path
import cv2
import xlsxwriter  # required dependency
import pytesseract
# Notes - OCR installation:
# 1. tesseract must to be installed <-- download from https://github.com/UB-Mannheim/tesseract/wiki
# 2. german training data must be added during installation
#    (can also manually be downloaded from https://github.com/tesseract-ocr/tessdata/blob/master/deu.traineddata
#    and be manually moved in folder: C:\ProgramData\Anaconda3\envs\*env_name*\Library\bin\tessdata)

# Set working directory to file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# %% Main
def main():
    # ***** 1) Scrape PDFs *****
    # parse the SNB website
    parser = 'html.parser'  # or 'lxml' (preferred) or 'html5lib', if installed
    main_url = 'https://www.snb.ch'
    resp = urllib.request.urlopen(main_url + '/en/iabout/monpol/id/monpol_current#')
    soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))

    # extract all links
    links = []
    for link in soup.find_all('a', href=True):
        links.append(link['href'])

    # only keep links with /en/mmr/reference/pre_
    links_lagebeurteilung = [main_url + s for s in links if "/en/mmr/reference/pre_" in s]

    # specify names of files
    names = []
    for i in links_lagebeurteilung:
        test = str(i.split('/')[-1])  # only take the last part of url for the name
        test = str(test.split('.')[-3])  # remove the points
        test = test + '.pdf'  # add pdf ending to name
        names.append(test)

    # specifiy download path
    pdf_dir = "SNB_Lagebeurteilungen/"

    # download the file to specific path
    def downloadFile(url, fileName):
        with open(pdf_dir + fileName, "wb") as file:
            response = requests.get(url)
            file.write(response.content)

    for idx, url in enumerate(links_lagebeurteilung):
        print("downloading {}".format(names[idx]))
        downloadFile(url, names[idx])

    # ***** 2) Read PDFs (~1h runtime) *****
    start = time.time()

    # i) specify cols for final df
    columns = ['date', 'text', 'pagenr', 'filename']
    data = []

    # ii) loop through pdf files
    for i, fname in enumerate(os.listdir(pdf_dir)):
        print("Extracting text from '{}' ({}/{})".format(fname, i + 1, len(os.listdir(pdf_dir))))
        path = pdf_dir + fname
        imgs = convert_from_path(path)

        # iii) loop page-images and apply OCR
        for i, img in enumerate(imgs):
            # scale images to increase OCR accuracy
            height, width = img.size
            img = np.array(img)
            img = cv2.resize(img, (3 * width, 3 * height), interpolation=cv2.INTER_LINEAR)
            img = cv2.bitwise_not(img)

            text = pytesseract.image_to_string(img, lang='deu')  # requires installation as specified at start of file
            date_str = fname[4:12]  # cut out 8 digit date
            date = datetime.datetime.strptime(date_str, '%Y%m%d')
            data.append([date, text, i, fname])

    # save final df
    df = pd.DataFrame(data, columns=columns)
    df.to_excel('articles_raw.xlsx', engine='xlsxwriter')

    print(
        "Saved text in DataFrame. Elapsed time: {}".format(time.strftime("%Mm %Ss", time.gmtime(time.time() - start))))


# %% Run file
if __name__ == '__main__':
    main()
