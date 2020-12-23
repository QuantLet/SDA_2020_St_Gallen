# Sentiment Analysis of Central Bank Monetary Policy Assessments

A central bank's communication is nowadays a core element of monetary policy, and in that sense it is actually a monetary policy tool itself. The analysis of central bank communication can provide important insights into the economic outlook as well as signals about the future course of policy.
In this repository, the swiss national Bank's monetary policy assessments are analyzed to predict developments in the economy.

## Repository structure

### 01_SNB_PDF_Webscaper
This script scrapes the [monetary policy assessment](https://www.snb.ch/en/iabout/monpol/id/monpol_current) pdf files of the Swiss National Bank from the period 2000 to 2020 and saves them in the data folder.

### 02_PDF_Reader
This script reads the pdf files from the folder data/lagebeurteilungenSNB with the package pytesseract. The output is an excel file called articles_raw.xlsx and saved in data folder.

### 03_Cleaning_EDA
This scipt cleans the unstrucuted text data and performs an LDA analysis.<br/>
The input is the articles_raw.xlsx taken from the data folder.
The output generates the following:
- articles_clean.xlsx in data folder
- Several plots saved in the plots folder (e.g 10 most common words, wordcloud)
- LDA_visualization.html

#### 10 most common words used in the monetary policy assessment
![plot1](../SDA_2020_St_Gallen/SDA_2020_St_Gallen_SentimentAnalysis_of_CentralBank_Statements/plots/most_common_words.png)

#### Wordcloud
![plot2](../SDA_2020_St_Gallen/SDA_2020_St_Gallen_SentimentAnalysis_of_CentralBank_Statements/plots/wordcloud.png)

#### LDA Visualization
![plot3](../SDA_2020_St_Gallen/SDA_2020_St_Gallen_SentimentAnalysis_of_CentralBank_Statements/plots/LDA_Visualization.png)


### 04_Sentiment
This script generates a sentiment indicator which counts the number of positive and negative words used in the monetary policy assessments. The fed dictionary was the main dictionary used in the analysis.
The two inputs are taken from the data folder:
- Dictionary.py (includes the [Fed dictionary](https://www.federalreserve.gov/econres/notes/ifdp-notes/constructing-a-dictionary-for-financial-stability-20170623.htm) and the [Lougrahn Mcdonald sentiment word list](https://sraf.nd.edu/textual-analysis/resources/))
- articles_clean.xlsx <br/>

The output is the following: 
- NetSentiment.csv timeseries indicator saved in data folder
- Plot of net positve vs. negative words


#### Count of positve vs. negative words with fed dictionary
![plot4](../SDA_2020_St_Gallen/SDA_2020_St_Gallen_SentimentAnalysis_of_CentralBank_Statements/plots/count_words.png)

### 05_Forecasting
This script uses economic data from [FRED](https://fred.stlouisfed.org/) and trains different models to predict the GDP growth in Switzerland. Explanatory variables are several macroeconomic variables and the NetSentiment.csv indicator.

#### Economic predictor variables for the GDP forecasting model
![plot5](../SDA_2020_St_Gallen/SDA_2020_St_Gallen_SentimentAnalysis_of_CentralBank_Statements/plots/economic_varibales.png)

#### Model results
- OLS (results: MSE: 2.83*e-05 and R2: 0.498)
- ElsticNet (MSE: 3.73*e-05 and R2: 0.338)
- RandomForest (MSE: 5.05*e-05 and R2: 0.103)
![plot6](../SDA_2020_St_Gallen/SDA_2020_St_Gallen_SentimentAnalysis_of_CentralBank_Statements/plots/model_results.png)
