Marketing Proposal 

[Marketing Campaign](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign) 



### Project Objectives:
- **Find a Problem**: Identify a significant issue that can be addressed with ML.
    - **Increase the effeciency** of the upcoming marketing campaign by targeting clients that are more likely to be receptive to the offer. Our client has a limited budget and can not have a dialing campaign for all customers and stay within budget. They need to know what customer are mostlikely to be receptive to the marketing campaign so they can maximize the value of there maketing campaign. 

- **Dataset Requirements**: Utilize a dataset with at least 500 records. For decision tree/random forest models, at least 1,000 records are required.
    - **Marketing Campaign has over 2000 records. 

### Project Element 

- **ML Model**: Implement either a supervised or unsupervised model.
    - **unsupervised**: Thinking we can use data to create groups that are likely to responde to specific campaigns. If not we can use supervised an look for clients likely to respond in a positive way. 
- **Evaluation**: Assess your model using testing data, incorporating necessary metrics and visualizations.
    - **80/20 split**

- **Technologies**: Mandatory use of Scikit-learn and at least three of the following:
  - API requests
  - Matplotlib
  - Pandas
  - Pandas plotting
  - Prophet
  - Python
  - Time series analysis



# About the Dataset

## Context
A response model can significantly enhance the efficiency of a marketing campaign by increasing responses or reducing expenses. The objective is to predict who will respond to an offer for a product or service.

## Content
- **AcceptedCmp1**: 1 if customer accepted the offer in the 1st campaign, 0 otherwise.
- **AcceptedCmp2**: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise.
- **AcceptedCmp3**: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise.
- **AcceptedCmp4**: 1 if customer accepted the offer in the 4th campaign, 0 otherwise.
- **AcceptedCmp5**: 1 if customer accepted the offer in the 5th campaign, 0 otherwise.
- **Response (target)**: 1 if customer accepted the offer in the last campaign, 0 otherwise.
- **Complain**: 1 if customer complained in the last 2 years.
- **DtCustomer**: Date of customer’s enrollment with the company.
- **Education**: Customer’s level of education.
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


