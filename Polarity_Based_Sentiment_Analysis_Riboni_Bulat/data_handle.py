#=============================================================================
#=============================================================================
#             BTC and ETH Predictions using Reddit Submissions
#                          Smart Data Analytics
#                         Universität St. Gallen
#
#                        Daniil Bulat 14-607-055
#                         Luca Riboni 14-619-878
#
#=============================================================================
#=============================================================================


#============================= Import Packages ===============================
import re
import pandas as pd
import numpy as np
import yfinance as yf
import os
import collections
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline
from statistics import mean
import statsmodels.api as sm


os.chdir("/Users/danielbulat/Desktop/Uni/HS20/SmartDataAnalytics/Smart_Data_Analytics_HS20/data")
 
#=============================== Functions ===================================
def IsNotNull(value):
	    return value is not None and len(value) > 0

 
def negated(word):

    if word.lower() in negate:
        return True
    else:
        return False
 
 
def polarity_check(dict, article):
  
    pos_count = 0
    neg_count = 0
 
    pos_words = []
    neg_words = []
 
    input_words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', article.lower())
 
    word_count = len(input_words)
 
    for i in range(0, word_count):
        if input_words[i] in dict['Negative']:
            neg_count += 1
            neg_words.append(input_words[i])
        if input_words[i] in dict['Positive']:
            if i >= 3:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]) or negated(input_words[i - 3]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 2:
                if negated(input_words[i - 1]) or negated(input_words[i - 2]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 1:
                if negated(input_words[i - 1]):
                    neg_count += 1
                    neg_words.append(input_words[i] + ' (with negation)')
                else:
                    pos_count += 1
                    pos_words.append(input_words[i])
            elif i == 0:
                pos_count += 1
                pos_words.append(input_words[i])
 
    print('The # of positive words:', pos_count)
    print('The # of negative words:', neg_count)
    print('The list of found positive words:', pos_words)
    print('The list of found negative words:', neg_words)
    polarity_score = (pos_count - neg_count) /(pos_count + neg_count)
    print(polarity_score)
    results = [polarity_score],pos_words,neg_words 
    
    return results

def data_handle(data, currency, df_currency):
    df_currency = pd.read_csv(data)
    df_currency['Full text'] = df_currency["Title"]+ ". " + df_currency["Text"].map(str)
    df_currency = np.array(df_currency)

    scores = []
    for i in df_currency[:,7]:
        try:
            scores.append(polarity_check(dictionary,i)[0])

        except:
            scores.append("0")
            pass
    df_currency = pd.DataFrame(df_currency)
    df_currency.columns =['Post ID', 'Title',"Text", "Score", "Publish Date", "al No. of Comments", "url", "Full Text" ] 
    df_currency["Polarity score"] = pd.DataFrame(scores)
    df_currency["Polarity score"] = df_currency["Polarity score"].astype(float)
    df_currency['Publish Date'] = df_currency['Publish Date'].astype('datetime64[ns]')



    dayly_mean_currency = df_currency.resample('d', on='Publish Date').mean().dropna(how='all')
    monthly_mean_currency = df_currency.resample('m', on='Publish Date').mean().dropna(how='all')
    weekly_mean_currency = pd.DataFrame(df_currency.groupby(df_currency["Publish Date"].dt.weekofyear)['Polarity score'].mean())
    print (dayly_mean_currency)
    print (monthly_mean_currency)
    print(weekly_mean_currency)
    results= df_currency, dayly_mean_currency, monthly_mean_currency, weekly_mean_currency
    return results

def clouds(data):
    text = " ".join(review for review in data["words"])

# Create and generate a word cloud image:
    stopwords_ = set(STOPWORDS)
    stopwords_.update(["use","game","Please","nan"])

    wordcloud = WordCloud(collocations=False, stopwords=stopwords_, background_color="white").generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    a=plt.show()
    return a



def count_graph(title,data):
    counts_ = collections.Counter(data["words"])
    clean_ = pd.DataFrame(counts_.most_common(15),
                             columns=['words', 'count'])
    fig, ax = plt.subplots(figsize=(8, 8))

    clean_.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

    ax.set_title("Common Words Found on Reddit"+str(title))

    a=plt.show()
    return a



#================================= Dictionary ================================
## Positive Words Dictionary
dict_p = []
f = open('lmhdictp.txt', 'r')
for line in f:
	t = line.strip().lower()
	if IsNotNull(t):
		dict_p.append(t)
f.close

## Negative Words Dictionary
dict_n = []
f = open('lmhdictn.txt', 'r')
for line in f:
	t = line.strip().lower()
	if IsNotNull(t):
		dict_n.append(t)
f.close

## Complete Dictionary
dictionary = {'Negative': dict_n,
         'Positive': dict_p}

negate = []
f = open('negate.txt', 'r')
for line in f:
	t = line.strip().lower()
	if IsNotNull(t):
		negate.append(t)




#============================ Sentiment Analysis =============================
df_BTC, daily_mean_BTC, monthly_mean_BTC, weekly_mean_BTC = data_handle("reddit_bitcoin.csv", "BTC", "df_BTC")
df_ETH, daily_mean_ETH, monthly_mean_ETH, weekly_mean_ETH = data_handle("reddit_eth.csv", "ETH", "df_ETH")

btc = yf.download("BTC-USD", start="2015-01-02", end="2020-12-02")
eth = yf.download("ETH-USD", start="2015-01-02", end="2020-12-02")

btc=btc.drop(['Open', 'High',"Low","Adj Close", "Volume"], axis=1)
eth=eth.drop(['Open', 'High',"Low","Adj Close", "Volume"], axis=1)

btc["Publish Date"]=btc.index
eth["Publish Date"]=eth.index
daily_mean_BTC['Publish Date'] = daily_mean_BTC.index
daily_mean_ETH['Publish Date'] = daily_mean_ETH.index

r = pd.date_range(start=daily_mean_BTC["Publish Date"].min(), end=daily_mean_BTC["Publish Date"].max())
daily_mean_ETH=daily_mean_ETH.set_index('Publish Date').reindex(r).fillna(0).rename_axis('Publish Date').reset_index()
eth=eth.set_index('Publish Date').reindex(r).fillna(0).rename_axis('Publish Date').reset_index()
daily_mean_BTC=daily_mean_BTC.set_index('Publish Date')
daily_mean_ETH=daily_mean_ETH.set_index('Publish Date')
btc=btc.set_index('Publish Date')
eth=eth.set_index('Publish Date')

daily_returns = pd.DataFrame([])
daily_returns["Publish Date"]= btc.index
daily_returns=daily_returns.set_index('Publish Date')
daily_returns["BTC polarity score"]=daily_mean_BTC["Polarity score"]
daily_returns["ETH polarity score"]=daily_mean_ETH["Polarity score"]
daily_returns["BTC"]=btc["Close"]
daily_returns["ETH"]=eth["Close"]



#============================== Word Clouds ===================================
## BTC
btc_text = " ".join(review for review in df_BTC["Full Text"])

# Create and generate a word cloud image:
btc_stopwords = set(STOPWORDS)
btc_stopwords.update(["New","will","way","use","one","nan","https","removed","deleted"])

btc_wordcloud = WordCloud(collocations=False, stopwords=btc_stopwords, background_color="white").generate(btc_text)

plt.imshow(btc_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.close()

## ETH
eth_text = " ".join(review for review in df_ETH["Full Text"])

# Create and generate a word cloud image:
eth_stopwords = set(STOPWORDS)
eth_stopwords.update(["use","game","Please","nan","removed","deleted","https"])

eth_wordcloud = WordCloud(collocations=False, stopwords=eth_stopwords, background_color="white").generate(eth_text)

plt.imshow(eth_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.close()

#graphs
ax = plt.gca()

daily_returns.plot(y="ETH polarity score")

plt.show()
ax = plt.gca()

dayly_returns.plot(y="BTC polarity score")

plt.show()
ax = plt.gca()

dayly_returns.plot(y="BTC")

plt.show()
ax = plt.gca()

dayly_returns.plot(y="ETH")

plt.show()
btc_part1=df_BTC.iloc[0:100000, 7]
btc_part2=df_BTC.iloc[100000:180000, 7]
btc_part3=df_BTC.iloc[180000:254526, 7]

df_ETH["Full Text"].to_csv('eth_fulltext.txt', header=True, index=False, sep='\t', mode='w')
df_BTC["Full Text"].to_csv('btc_fulltext.txt', header=True, index=False, sep='\t', mode='w')

btc_part1.to_csv('btc_fulltext1.txt', header=True, index=False, sep='\t', mode='w')
btc_part2.to_csv('btc_fulltext2.txt', header=True, index=False, sep='\t', mode='w')
btc_part3.to_csv('btc_fulltext3.txt', header=True, index=False, sep='\t', mode='w')


with open("eth_fulltext.txt", "r", encoding ="utf8") as current_file: 
        text_eth = current_file.read() 
        text_eth = text_eth.replace("\n", "").replace("\r", "") 

with open("btc_fulltext1.txt", "r", encoding ="utf8") as current_file: 
        text_btc1 = current_file.read() 
        text_btc1 = text_btc1.replace("\n", "").replace("\r", "") 
with open("btc_fulltext2.txt", "r", encoding ="utf8") as current_file: 
        text_btc2 = current_file.read() 
        text_btc2 = text_btc2.replace("\n", "").replace("\r", "") 
with open("btc_fulltext3.txt", "r", encoding ="utf8") as current_file: 
        text_btc3 = current_file.read() 
        text_btc3 = text_btc3.replace("\n", "").replace("\r", "") 

pos_eth=polarity_check(dictionary,text_eth)[1]
neg_eth=polarity_check(dictionary,text_eth)[2]

pos_btc1=polarity_check(dictionary,text_btc1)[1]
neg_btc1=polarity_check(dictionary,text_btc1)[2]
pos_btc2=polarity_check(dictionary,text_btc2)[1]
neg_btc2=polarity_check(dictionary,text_btc2)[2]
pos_btc3=polarity_check(dictionary,text_btc3)[1]
neg_btc3=polarity_check(dictionary,text_btc3)[2]

pos_btc = pos_btc1 +pos_btc2 +pos_btc3
neg_btc = neg_btc1 +neg_btc2 +neg_btc3

pos_btc =pd.DataFrame(pos_btc)
neg_btc=pd.DataFrame(neg_btc)
pos_eth =pd.DataFrame(pos_eth)
neg_eth=pd.DataFrame(neg_eth)

pos_btc.columns=["words"]
neg_btc.columns=["words"]
pos_eth.columns=["words"]
neg_eth.columns=["words"]

clouds(pos_btc)
clouds(neg_btc)
clouds(pos_eth)
clouds(neg_eth)

count_graph(" (Positive, Bitcoin)",pos_btc)
count_graph(" (Negative, Bitcoin)",neg_btc)
count_graph(" (Positive, Ethereum)",pos_eth)
count_graph(" (Negative, Ethereum)",neg_eth)



#================================= OLS =======================================
### Bitcoin
n = len(btc["Close"])

btc_ols_var = pd.DataFrame([])
btc_ols_var["daily score"] = daily_mean_BTC["Polarity score"]

## Weekly Time Series of Polarity Scores
weekly_ts = np.zeros([n,1])
for i in range(0,n-7):
  weekly_ts[i+7] = mean(btc_ols_var["daily score"][i:i+7])

btc_ols_var["weekly score"] = weekly_ts

## Monthly Time Series of Polarity Scores
monthly_ts = np.zeros([n,1])
for i in range(0,n-30):
  monthly_ts[i+30] = mean(btc_ols_var["daily score"][i:i+30])

btc_ols_var["monthly score"] = monthly_ts

## Add constant
btc_ols_var = sm.add_constant(btc_ols_var)

btc_ols_var = btc_ols_var.iloc[30:,]
y_btc = btc["Close"].pct_change()
y_btc = y_btc.iloc[30:,]


## OLS Model
results = sm.OLS(y_btc, btc_ols_var).fit()
print(results.summary())


### Ethereum
n = len(eth["Close"])

eth_ols_var = pd.DataFrame([])
eth_ols_var["daily score"] = daily_mean_ETH["Polarity score"]

## Weekly Time Series of Polarity Scores
weekly_ts = np.zeros([n,1])
for i in range(0,n-7):
  weekly_ts[i+7] = mean(eth_ols_var["daily score"][i:i+7])

eth_ols_var["weekly score"] = weekly_ts

## Monthly Time Series of Polarity Scores
monthly_ts = np.zeros([n,1])
for i in range(0,n-30):
  monthly_ts[i+30] = mean(eth_ols_var["daily score"][i:i+30])

eth_ols_var["monthly score"] = monthly_ts

## Add constant
eth_ols_var = sm.add_constant(eth_ols_var)

eth_ols_var = eth_ols_var.iloc[219:,]
y_eth = eth["Close"].pct_change()
y_eth = y_eth.iloc[219:,]


## OLS Model
results = sm.OLS(y_eth, eth_ols_var).fit()
print(results.summary())












