from bs4 import BeautifulSoup
import urllib.request
import requests

#%% Main
def main():
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
    pdf_dir = "../SDA_2020_St_Gallen/SDA_2020_St_Gallen_SentimentAnalysis_of_CentralBank_Statements/data/lagebeurteilungenSNB/"

    # download the file to specific path
    def downloadFile(url, fileName):
        with open(pdf_dir + fileName, "wb") as file:
            response = requests.get(url)
            file.write(response.content)

    for idx, url in enumerate(links_lagebeurteilung):
        print("downloading {}".format(names[idx]))
        downloadFile(url, names[idx])

# %% Run file
if __name__ == '__main__':
    main()