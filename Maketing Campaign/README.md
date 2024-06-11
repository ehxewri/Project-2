Marketing Proposal 

[Marketing Campaign](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign) 



### Project Objectives:
- ** Define a Problem**: Identify a significant issue that can be addressed with ML.
    - **Increase the effeciency** of the upcoming marketing campaign by targeting clients that are more likely to be receptive to the offer. Our client has a limited budget and can not have a dialing campaign for all customers and stay within budget. They need to know what customer are mostlikely to be receptive to the marketing campaign so they can maximize the value of there maketing campaign. 

- **Dataset Requirements**: (Why did we pick this project and this dataset?)
Utilize a dataset with at least 500 records. For decision tree/random forest models, at least 1,000 records are required.
    - ** Marketing Campaign has over 2000 records. 

### Project Element 

- **ML Model**: Implement either a supervised or unsupervised model.
    - **supervised**: The business wanted to know how likely customers are to respond to a new campaignin a positive way. To do this we need to look at what we know about our clients and how they have responded in the past. Our df - response is our X. y will be the repsonce column. 

- **Evaluation**: Assess your model using testing data, incorporating necessary metrics and visualizations.

    Describe each field quickly

    Describe the data pre processing we need to do. 



    - **80/20 split**
        Why we slit before preprocessing? Add verbage. 

    - **Standard Scaler** Testing shows that standard scaler has a lower training score but a much better test score than Min/Max. We will use Standard Scaler
    - Balance date
    - Encoding
        - eduction
        - Maital status
        - generation
        - How long has this cuatomer been with us in months
        why did we drop na
        why did we drop index

        Frank to add coorilation and recommend droping columns
    
    What model do we use that best suite our data and why

    # Test models
        -RANDOM FOREST MODEL *
        -GradientBoostingClassifier
        -KNeighborsClassifier
        -SVM (Support Vector Machine)
        -LogisticRegression
        -Decision Tree Model *


- **Technologies**: Mandatory use of Scikit-learn and at least three of the following:
  - API requests - not needed. we imported this using an API anyway so anyone who has a kaggle account can setup and run this. 
  - Matplotlib - this will be good for a scaterplat 
  - Pandas - Marketing_Campaign.ipynb 
  - Pandas plotting - we can use this for picking the best n-cluster
  - Prophet
  - Python - for creatign functions that simplify the code
  - Time series analysis - we will not use time series for this dataset



# About the Dataset

## Context
A response model can significantly enhance the efficiency of a marketing campaign by increasing responses or reducing expenses. The objective is to predict who will respond to an offer for a product or service.

## Content - Data exploration

Response and accept columns assign 0 to no reponse. 1 to 5 attempt 5 to first attempt. the idea is the most favorable response would have the highest value to the lowest value being no responce
_____________________________________________
- **AcceptedCmp1**: 1 if customer accepted the offer in the 1st campaign, 0 otherwise.
- **AcceptedCmp2**: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise.
- **AcceptedCmp3**: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise.
- **AcceptedCmp4**: 1 if customer accepted the offer in the 4th campaign, 0 otherwise.
- **AcceptedCmp5**: 1 if customer accepted the offer in the 5th campaign, 0 otherwise.
- **Response (target)**: 1 if customer accepted the offer in the last campaign, 0 otherwise.
____________________________________________
- **Complain**: 1 if customer complained in the last 2 years.

- **DtCustomer**: Date of customer’s enrollment with the company. we should look to convert to months


For year of birth I want to try something new. Based on generational values, group ages into 
age ranges.Generational groupings provide a way to categorize cohorts of people who experienced similar societal events during formative years. These generational labels are primarily applied in the United States and other Western countries and can vary somewhat by definition and cultural interpretation. Here's a broad overview of the generations from the Greatest Generation to those being born today:

1. The Greatest Generation
Birth Years: Early 1900s to 1927
Age Range (as of 2023): 96 years and older
Characteristics: Grew up during the Great Depression, many fought in World War II.
2. The Silent Generation
Birth Years: 1928 to 1945
Age Range (as of 2023): 78 to 95 years
Characteristics: Known for hard work and strong values, often seen as the traditionalists who played a key role in shaping post-war prosperity.
3. Baby Boomers
Birth Years: 1946 to 1964
Age Range (as of 2023): 59 to 77 years
Characteristics: Experienced dramatic social changes, economic prosperity, and the civil rights movement. Known for being work-centric.
4. Generation X
Birth Years: 1965 to 1980
Age Range (as of 2023): 43 to 58 years
Characteristics: Often referred to as the "latchkey" generation, valued independence, and work-life balance, experienced the rise of personal computing.
5. Millennials (Gen Y)
Birth Years: 1981 to 1996
Age Range (as of 2023): 27 to 42 years
Characteristics: Tech-savvy, value flexibility and purpose in their work, experienced the growth of the internet and social media.
6. Generation Z (Gen Z)
Birth Years: 1997 to 2012
Age Range (as of 2023): 11 to 26 years
Characteristics: Digital natives who value individual expression and avoid labels, highly connected, concerned with mental health and the environment.
7. Generation Alpha
Birth Years: 2013 to present (as of 2023, the end year for this generation has not been fully defined yet)
Age Range (as of 2023): 0 to 10 years
Characteristics: Very young and still being shaped, but expected to be the most technologically immersed.

_________________________________________________
- **Year_Birth**: year customer was born.
_________________________________________________


set education from 0 for none to highest level.
- **Education**: 
    - **Basic** This generally refers to elementary or primary education.
    - **2n Cycle** This is not a commonly used term globally but might refer to secondary education or an intermediary level in some education systems.
    - **Graduation** Typically refers to the completion of a bachelor's or undergraduate degree.
    - **Master** A postgraduate degree that follows the completion of a bachelor's degree.
    - **PhD** The highest university degree, typically following a master's degree.

- **Marital**: Customer’s marital status.
- **Kidhome**: Number of small children in customer’s household.
- **Teenhome**: Number of teenagers in customer’s household.
- **Income**: Customer’s yearly household income.
- **MntFishProducts**: Amount spent on fish products in the last 2 years.
- **MntMeatProducts**: Amount spent on meat products in the last 2 years.
- **MntFruits**: Amount spent on fruit products in the last 2 years.
- **MntSweetProducts**: Amount spent on sweet products in the last 2 years.
- **MntWines**: Amount spent on wine products in the last 2 years.
- **MntGoldProds**: Amount spent on gold products in the last 2 years.
- **NumDealsPurchases**: Number of purchases made with a discount.
- **NumCatalogPurchases**: Number of purchases made using a catalogue.
- **NumStorePurchases**: Number of purchases made directly in stores.
- **NumWebPurchases**: Number of purchases made through the company’s website.
- **NumWebVisitsMonth**: Number of visits to the company’s website in the last month.
- **Recency**: Number of days since the last purchase.

## Acknowledgements
O. Parr-Rud. *Business Analytics Using SAS Enterprise Guide and SAS Enterprise Miner.* SAS Institute, 2014.

## Inspiration
The main objective is to train a predictive model which allows the company to maximize the profit of the next marketing campaign.


# Pre ipynb execute setup


## Install and Set Up Kaggle and API Key

Follow these steps to install and configure the Kaggle API on your system:

1. **Create a Kaggle Account**
   - Visit [Kaggle](https://www.kaggle.com) and sign up for an account.

2. **Obtain Kaggle API Key**
   - Go to your Kaggle account settings.
   - Find the "API" section and click on "Create New API Token".
   - This will download a `kaggle.json` file containing your API key.

3. **Install Kaggle Package**
   - Use Conda to install the Kaggle package by running:
     ```bash
     conda install kaggle
     ```
    Note: this should download the json with you key to downloads


4. **Configure API Key**
   - Copy the `kaggle.json` file to your user directory under the `.kaggle` folder. On most systems, you can use the following command:

To use the API to import a dataset use

  kaggle.api.dataset_download_files('rodsaldanha/arketing-campaign', path='resources', unzip=True)

    -   Path this will put the files in a directory. if none is specified it looks in your current working directory. if the directory is not found it will create it. 
    -   unzip: 

#


Initial Analysis

NA fields
