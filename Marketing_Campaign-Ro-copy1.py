{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Install and setup kaggle and api key\n",
    "get a kaggle account\n",
    "Get Kaggle API\n",
    "conda install kaggle\n",
    "copy the kaggle.json file from download to your user directory/.kaggle\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "# pip install kaggle\n",
    "import os\n",
    "os.environ['KAGGLE_USERNAME'] = 'romarioesparza'\n",
    "os.environ['KAGGLE_KEY'] = '2ac298547b3f07e56d1f297644fd12de'\n",
    "\n",
    "import kaggle\n",
    "\n",
    "\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "import zipfile\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset URL: https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-02 21:49:38,369 WARNING Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))': /api/v1/datasets/download/rodsaldanha/arketing-campaign?datasetVersionNumber=None\n"
     ]
    }
   ],
   "source": [
    "# Get the data using an API call\n",
    "kaggle.api.dataset_download_files('rodsaldanha/arketing-campaign', path='resources', unzip=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the data\n",
    "data = pd.read_csv(\"./resources/marketing_campaign.csv\",delimiter=';')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# EDA (Exploratory Data Analysis)\n",
    "We will revisit this. For now We want the rough draft of the model\n",
    "#\n",
    "During EDA\n",
    "\n",
    "Visualize the data using plots and graphs to understand distributions and relationships between variables.\n",
    "Calculate summary statistics to get a sense of the central tendencies and variability.\n",
    "Identify any correlations between variables that might influence model choices.\n",
    "Detect and treat missing values or outliers that could skew the results of your analysis.\n",
    "Explore the data's structure to inform feature selection and engineering, which are key to building effective machine learning models.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Year_Birth</th>\n",
       "      <th>Education</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Income</th>\n",
       "      <th>Kidhome</th>\n",
       "      <th>Teenhome</th>\n",
       "      <th>Dt_Customer</th>\n",
       "      <th>Recency</th>\n",
       "      <th>MntWines</th>\n",
       "      <th>...</th>\n",
       "      <th>NumWebVisitsMonth</th>\n",
       "      <th>AcceptedCmp3</th>\n",
       "      <th>AcceptedCmp4</th>\n",
       "      <th>AcceptedCmp5</th>\n",
       "      <th>AcceptedCmp1</th>\n",
       "      <th>AcceptedCmp2</th>\n",
       "      <th>Complain</th>\n",
       "      <th>Z_CostContact</th>\n",
       "      <th>Z_Revenue</th>\n",
       "      <th>Response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5524</td>\n",
       "      <td>1957</td>\n",
       "      <td>Graduation</td>\n",
       "      <td>Single</td>\n",
       "      <td>58138.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2012-09-04</td>\n",
       "      <td>58</td>\n",
       "      <td>635</td>\n",
       "      <td>...</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2174</td>\n",
       "      <td>1954</td>\n",
       "      <td>Graduation</td>\n",
       "      <td>Single</td>\n",
       "      <td>46344.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2014-03-08</td>\n",
       "      <td>38</td>\n",
       "      <td>11</td>\n",
       "      <td>...</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4141</td>\n",
       "      <td>1965</td>\n",
       "      <td>Graduation</td>\n",
       "      <td>Together</td>\n",
       "      <td>71613.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2013-08-21</td>\n",
       "      <td>26</td>\n",
       "      <td>426</td>\n",
       "      <td>...</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6182</td>\n",
       "      <td>1984</td>\n",
       "      <td>Graduation</td>\n",
       "      <td>Together</td>\n",
       "      <td>26646.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2014-02-10</td>\n",
       "      <td>26</td>\n",
       "      <td>11</td>\n",
       "      <td>...</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5324</td>\n",
       "      <td>1981</td>\n",
       "      <td>PhD</td>\n",
       "      <td>Married</td>\n",
       "      <td>58293.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2014-01-19</td>\n",
       "      <td>94</td>\n",
       "      <td>173</td>\n",
       "      <td>...</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 29 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     ID  Year_Birth   Education Marital_Status   Income  Kidhome  Teenhome  \\\n",
       "0  5524        1957  Graduation         Single  58138.0        0         0   \n",
       "1  2174        1954  Graduation         Single  46344.0        1         1   \n",
       "2  4141        1965  Graduation       Together  71613.0        0         0   \n",
       "3  6182        1984  Graduation       Together  26646.0        1         0   \n",
       "4  5324        1981         PhD        Married  58293.0        1         0   \n",
       "\n",
       "  Dt_Customer  Recency  MntWines  ...  NumWebVisitsMonth  AcceptedCmp3  \\\n",
       "0  2012-09-04       58       635  ...                  7             0   \n",
       "1  2014-03-08       38        11  ...                  5             0   \n",
       "2  2013-08-21       26       426  ...                  4             0   \n",
       "3  2014-02-10       26        11  ...                  6             0   \n",
       "4  2014-01-19       94       173  ...                  5             0   \n",
       "\n",
       "   AcceptedCmp4  AcceptedCmp5  AcceptedCmp1  AcceptedCmp2  Complain  \\\n",
       "0             0             0             0             0         0   \n",
       "1             0             0             0             0         0   \n",
       "2             0             0             0             0         0   \n",
       "3             0             0             0             0         0   \n",
       "4             0             0             0             0         0   \n",
       "\n",
       "   Z_CostContact  Z_Revenue  Response  \n",
       "0              3         11         1  \n",
       "1              3         11         0  \n",
       "2              3         11         0  \n",
       "3              3         11         0  \n",
       "4              3         11         0  \n",
       "\n",
       "[5 rows x 29 columns]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Columns with NA valuses \n",
      " Income    24\n",
      "dtype: int64\n",
      "Columns that are not numeric :\n",
      " ['Education', 'Marital_Status', 'Dt_Customer']\n"
     ]
    }
   ],
   "source": [
    "display (data.head())\n",
    "# what does our data look like? At this point also use any documentation on the data set to find out what each value means and how it might be used is solving the business problem\n",
    "print (f'Columns with NA valuses \\n {data.isna().sum()[lambda x: x > 0]}')\n",
    "# Make desision about null values. Can we fill them of should we drop rows with null values?\n",
    "non_numeric= (data.dtypes[(data.dtypes != 'int64') & (data.dtypes != 'float64')]).index.tolist()\n",
    "print (f'Columns that are not numeric :\\n {non_numeric}')\n",
    "# Explore non numberic type to see how we can use them in the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Year_Birth</th>\n",
       "      <th>Income</th>\n",
       "      <th>Kidhome</th>\n",
       "      <th>Teenhome</th>\n",
       "      <th>Recency</th>\n",
       "      <th>MntWines</th>\n",
       "      <th>MntFruits</th>\n",
       "      <th>MntMeatProducts</th>\n",
       "      <th>MntFishProducts</th>\n",
       "      <th>...</th>\n",
       "      <th>NumWebVisitsMonth</th>\n",
       "      <th>AcceptedCmp3</th>\n",
       "      <th>AcceptedCmp4</th>\n",
       "      <th>AcceptedCmp5</th>\n",
       "      <th>AcceptedCmp1</th>\n",
       "      <th>AcceptedCmp2</th>\n",
       "      <th>Complain</th>\n",
       "      <th>Z_CostContact</th>\n",
       "      <th>Z_Revenue</th>\n",
       "      <th>Response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5524</td>\n",
       "      <td>1957</td>\n",
       "      <td>58138.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>58</td>\n",
       "      <td>635</td>\n",
       "      <td>88</td>\n",
       "      <td>546</td>\n",
       "      <td>172</td>\n",
       "      <td>...</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2174</td>\n",
       "      <td>1954</td>\n",
       "      <td>46344.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38</td>\n",
       "      <td>11</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>2</td>\n",
       "      <td>...</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4141</td>\n",
       "      <td>1965</td>\n",
       "      <td>71613.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>26</td>\n",
       "      <td>426</td>\n",
       "      <td>49</td>\n",
       "      <td>127</td>\n",
       "      <td>111</td>\n",
       "      <td>...</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6182</td>\n",
       "      <td>1984</td>\n",
       "      <td>26646.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>26</td>\n",
       "      <td>11</td>\n",
       "      <td>4</td>\n",
       "      <td>20</td>\n",
       "      <td>10</td>\n",
       "      <td>...</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5324</td>\n",
       "      <td>1981</td>\n",
       "      <td>58293.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>94</td>\n",
       "      <td>173</td>\n",
       "      <td>43</td>\n",
       "      <td>118</td>\n",
       "      <td>46</td>\n",
       "      <td>...</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     ID  Year_Birth   Income  Kidhome  Teenhome  Recency  MntWines  MntFruits  \\\n",
       "0  5524        1957  58138.0        0         0       58       635         88   \n",
       "1  2174        1954  46344.0        1         1       38        11          1   \n",
       "2  4141        1965  71613.0        0         0       26       426         49   \n",
       "3  6182        1984  26646.0        1         0       26        11          4   \n",
       "4  5324        1981  58293.0        1         0       94       173         43   \n",
       "\n",
       "   MntMeatProducts  MntFishProducts  ...  NumWebVisitsMonth  AcceptedCmp3  \\\n",
       "0              546              172  ...                  7             0   \n",
       "1                6                2  ...                  5             0   \n",
       "2              127              111  ...                  4             0   \n",
       "3               20               10  ...                  6             0   \n",
       "4              118               46  ...                  5             0   \n",
       "\n",
       "   AcceptedCmp4  AcceptedCmp5  AcceptedCmp1  AcceptedCmp2  Complain  \\\n",
       "0             0             0             0             0         0   \n",
       "1             0             0             0             0         0   \n",
       "2             0             0             0             0         0   \n",
       "3             0             0             0             0         0   \n",
       "4             0             0             0             0         0   \n",
       "\n",
       "   Z_CostContact  Z_Revenue  Response  \n",
       "0              3         11         1  \n",
       "1              3         11         0  \n",
       "2              3         11         0  \n",
       "3              3         11         0  \n",
       "4              3         11         0  \n",
       "\n",
       "[5 rows x 26 columns]"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# just to get started we will drop NA and columns that are not numberic. this will let us get a rough model\n",
    "# we come back to this and preprocess based on the draft results if needed\n",
    "\n",
    "\n",
    "data_drop_columns = data.drop(columns=non_numeric, axis=1)\n",
    "data_drop_na = data_drop_columns.dropna()\n",
    "df = data_drop_na.copy()\n",
    "df.head()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data into Train and Test\n",
    "X = df.drop('Response', axis=1)\n",
    "y = df[\"Response\"]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)\n",
    "# This will split 'X' and 'y' such that 80% is used for training and 20% is used for testing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.13477191,  0.27476758, -0.55482932, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [-0.30937932, -2.03667158, -0.27404419, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [-1.1285252 ,  0.35731898,  0.67127093, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       ...,\n",
       "       [-1.64618844, -0.30309221,  0.55808112, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [-1.05184575,  0.27476758,  1.08717858, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [ 1.61376579, -0.880952  , -0.21188861, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ]])"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Scale the X data by using StandardScaler()\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
    "scaler_ss = StandardScaler().fit(X_train)\n",
    "X_train_ss_scaled = scaler_ss.transform(X_train)\n",
    "X_train_ss_scaled"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "what does a logistic model score look like without scaling?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "# #\n",
    "# logistic_regression_model = LogisticRegression()\n",
    "\n",
    "# # Fit the model\n",
    "# logistic_regression_model.fit(X_train_ss_scaled, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Scaling the data evens out the high and low values so on one column dominates over all others\n",
    "We will want to compare the scores of standard scalar to Min Max scalar to pick the bast scaling methood."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.13477191,  0.27476758, -0.55482932, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [-0.30937932, -2.03667158, -0.27404419, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [-1.1285252 ,  0.35731898,  0.67127093, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       ...,\n",
       "       [-1.64618844, -0.30309221,  0.55808112, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [-1.05184575,  0.27476758,  1.08717858, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [ 1.61376579, -0.880952  , -0.21188861, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ]])"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Scale the X data by using StandardScaler()\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler\n",
    "scaler_ss = StandardScaler().fit(X_train)\n",
    "X_train_ss_scaled = scaler_ss.transform(X_train)\n",
    "X_train_ss_scaled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.28465542,  1.67814135, -1.34041544, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [-0.33894248, -1.62391459,  0.41722354, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [-1.54518097, -0.38564361, -0.37026134, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       ...,\n",
       "       [-0.19420618,  0.43987037, -1.19440407, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [ 1.40128059,  0.60497317,  0.55792611, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ],\n",
       "       [ 1.63378252,  0.43987037,  0.59547522, ..., -0.0923974 ,\n",
       "         0.        ,  0.        ]])"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Transform the test dataset based on the fit from the training dataset\n",
    "X_test_ss_scaled = scaler_ss.transform(X_test)\n",
    "X_test_ss_scaled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" checked><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create a `LogisticRegression` function and assign it \n",
    "# to a variable named `logistic_regression_model`.\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "#\n",
    "logistic_regression_model_ss = LogisticRegression()\n",
    "\n",
    "# Fit the model\n",
    "logistic_regression_model_ss.fit(X_train_ss_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Data Score: 0.8797968397291196\n",
      "Testing Data Score: 0.8963963963963963\n"
     ]
    }
   ],
   "source": [
    "# Score the model\n",
    "print(f\"Training Data Score: {logistic_regression_model_ss.score(X_train_ss_scaled, y_train)}\")\n",
    "print(f\"Testing Data Score: {logistic_regression_model_ss.score(X_test_ss_scaled, y_test)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.46054866, 0.76699029, 0.05418567, ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       [0.40988294, 0.49514563, 0.06508296, ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       [0.17219194, 0.77669903, 0.1017707 , ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       ...,\n",
       "       [0.02198195, 0.69902913, 0.09737779, ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       [0.19444196, 0.76699029, 0.1179121 , ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       [0.96792065, 0.63106796, 0.06749522, ..., 0.        , 0.        ,\n",
       "        0.        ]])"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "array([[0.5822536 , 0.93203883, 0.02369702, ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       [0.40130462, 0.54368932, 0.0919111 , ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       [0.05129122, 0.68932039, 0.06134876, ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       ...,\n",
       "       [0.44330265, 0.78640777, 0.02936373, ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       [0.90626396, 0.80582524, 0.09737178, ..., 0.        , 0.        ,\n",
       "        0.        ],\n",
       "       [0.97372889, 0.78640777, 0.09882906, ..., 0.        , 0.        ,\n",
       "        0.        ]])"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# now lets look at min max scaler\n",
    "scaler_mm = MinMaxScaler().fit(X_train)\n",
    "X_train_mm_scaled = scaler_mm.transform(X_train)\n",
    "display (X_train_mm_scaled)\n",
    "\n",
    "X_test_mm_scaled = scaler_mm.transform(X_test)\n",
    "display (X_test_mm_scaled)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# I want to come back to this. Its importand to understand pre scaling \n",
    "logistic_regression_model_mm = LogisticRegression()\n",
    "\n",
    "# Fit the model\n",
    "logistic_regression_model_mm.fit(X_train_mm_scaled, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-5\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" checked><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_regression_model_mm = LogisticRegression()\n",
    "logistic_regression_model_mm.fit(X_train_mm_scaled, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Standard Scaler\n",
      "Training Data Score: 0.8797968397291196\n",
      "Testing Data Score: 0.8963963963963963\n",
      "Min Max Scaler\n",
      "Training Data Score: 0.8820541760722348\n",
      "Testing Data Score: 0.8986486486486487\n"
     ]
    }
   ],
   "source": [
    "# Score the model\n",
    "\n",
    "print(f\"Standard Scaler\\nTraining Data Score: {logistic_regression_model_ss.score(X_train_ss_scaled, y_train)}\")\n",
    "print(f\"Testing Data Score: {logistic_regression_model_ss.score(X_test_ss_scaled, y_test)}\")\n",
    "print(f\"Min Max Scaler\\nTraining Data Score: {logistic_regression_model_mm.score(X_train_mm_scaled, y_train)}\")\n",
    "print(f\"Testing Data Score: {logistic_regression_model_mm.score(X_test_mm_scaled, y_test)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ************** New Implementation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest - Training Data Score: 1.0\n",
      "Random Forest - Testing Data Score: 0.8896396396396397\n"
     ]
    }
   ],
   "source": [
    "# **RANDOM FOREST MODEL\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Create and train the model\n",
    "random_forest_model = RandomForestClassifier(random_state=1)\n",
    "random_forest_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Score the model\n",
    "print(f\"Random Forest - Training Data Score: {random_forest_model.score(X_train_ss_scaled, y_train)}\")\n",
    "print(f\"Random Forest - Testing Data Score: {random_forest_model.score(X_test_ss_scaled, y_test)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix\n",
    "from sklearn.model_selection import cross_val_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest - Precision: 0.7083333333333334\n",
      "Random Forest - Recall: 0.288135593220339\n",
      "Random Forest - F1 Score: 0.4096385542168675\n",
      "Random Forest - Cross-Validation Accuracy: 0.8690761518262115\n"
     ]
    }
   ],
   "source": [
    "# *******RANDOM FOREST WITH METRICS*********\n",
    "\n",
    "# Create and train the model\n",
    "random_forest_model = RandomForestClassifier(random_state=1)\n",
    "random_forest_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Predict on test set\n",
    "y_pred = random_forest_model.predict(X_test_ss_scaled)\n",
    "\n",
    "# Calculate precision, recall, F1 score\n",
    "print(f\"Random Forest - Precision: {precision_score(y_test, y_pred)}\")\n",
    "print(f\"Random Forest - Recall: {recall_score(y_test, y_pred)}\")\n",
    "print(f\"Random Forest - F1 Score: {f1_score(y_test, y_pred)}\")\n",
    "\n",
    "# Cross-validation scores\n",
    "cv_scores = cross_val_score(random_forest_model, X_train_ss_scaled, y_train, cv=5, scoring='accuracy')\n",
    "print(f\"Random Forest - Cross-Validation Accuracy: {cv_scores.mean()}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gradient Boosting Machine - Training Data Score: 0.9379232505643341\n",
      "Gradient Boosting Machine - Testing Data Score: 0.9009009009009009\n"
     ]
    }
   ],
   "source": [
    "# ******* GradientBoostingClassifier MODELING\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "\n",
    "# Create and train the model\n",
    "gbm_model = GradientBoostingClassifier(random_state=1)\n",
    "gbm_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Score the model\n",
    "print(f\"Gradient Boosting Machine - Training Data Score: {gbm_model.score(X_train_ss_scaled, y_train)}\")\n",
    "print(f\"Gradient Boosting Machine - Testing Data Score: {gbm_model.score(X_test_ss_scaled, y_test)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gradient Boosting Machine - Precision: 0.7142857142857143\n",
      "Gradient Boosting Machine - Recall: 0.423728813559322\n",
      "Gradient Boosting Machine - F1 Score: 0.5319148936170213\n",
      "Gradient Boosting Machine - Cross-Validation Accuracy: 0.8764064613670725\n"
     ]
    }
   ],
   "source": [
    "# ******* GradientBoostingClassifier MODELING WITH METRICS\n",
    "\n",
    "# Create and train the model\n",
    "gbm_model = GradientBoostingClassifier(random_state=1)\n",
    "gbm_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Predict on test set\n",
    "y_pred = gbm_model.predict(X_test_ss_scaled)\n",
    "\n",
    "# Calculate precision, recall, F1 score\n",
    "print(f\"Gradient Boosting Machine - Precision: {precision_score(y_test, y_pred)}\")\n",
    "print(f\"Gradient Boosting Machine - Recall: {recall_score(y_test, y_pred)}\")\n",
    "print(f\"Gradient Boosting Machine - F1 Score: {f1_score(y_test, y_pred)}\")\n",
    "\n",
    "# Cross-validation scores\n",
    "cv_scores = cross_val_score(gbm_model, X_train_ss_scaled, y_train, cv=5, scoring='accuracy')\n",
    "print(f\"Gradient Boosting Machine - Cross-Validation Accuracy: {cv_scores.mean()}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K-Nearest Neighbors - Training Data Score: 0.8955981941309256\n",
      "K-Nearest Neighbors - Testing Data Score: 0.8738738738738738\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# Create and train the model\n",
    "knn_model = KNeighborsClassifier()\n",
    "knn_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Score the model\n",
    "print(f\"K-Nearest Neighbors - Training Data Score: {knn_model.score(X_train_ss_scaled, y_train)}\")\n",
    "print(f\"K-Nearest Neighbors - Testing Data Score: {knn_model.score(X_test_ss_scaled, y_test)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K-Nearest Neighbors - Precision: 0.5384615384615384\n",
      "K-Nearest Neighbors - Recall: 0.3559322033898305\n",
      "K-Nearest Neighbors - F1 Score: 0.42857142857142855\n",
      "K-Nearest Neighbors - Cross-Validation Accuracy: 0.8634312087212541\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# Create and train the model\n",
    "knn_model = KNeighborsClassifier()\n",
    "knn_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Predict on test set\n",
    "y_pred = knn_model.predict(X_test_ss_scaled)\n",
    "\n",
    "# Calculate precision, recall, F1 score\n",
    "print(f\"K-Nearest Neighbors - Precision: {precision_score(y_test, y_pred)}\")\n",
    "print(f\"K-Nearest Neighbors - Recall: {recall_score(y_test, y_pred)}\")\n",
    "print(f\"K-Nearest Neighbors - F1 Score: {f1_score(y_test, y_pred)}\")\n",
    "\n",
    "# Cross-validation scores\n",
    "cv_scores = cross_val_score(knn_model, X_train_ss_scaled, y_train, cv=5, scoring='accuracy')\n",
    "print(f\"K-Nearest Neighbors - Cross-Validation Accuracy: {cv_scores.mean()}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Support Vector Machine - Training Data Score: 0.9091422121896162\n",
      "Support Vector Machine - Testing Data Score: 0.8851351351351351\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "\n",
    "# Create and train the model\n",
    "svm_model = SVC()\n",
    "svm_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Score the model\n",
    "print(f\"Support Vector Machine - Training Data Score: {svm_model.score(X_train_ss_scaled, y_train)}\")\n",
    "print(f\"Support Vector Machine - Testing Data Score: {svm_model.score(X_test_ss_scaled, y_test)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Support Vector Machine - Precision: 0.6666666666666666\n",
      "Support Vector Machine - Recall: 0.2711864406779661\n",
      "Support Vector Machine - F1 Score: 0.38554216867469876\n",
      "Support Vector Machine - Cross-Validation Accuracy: 0.8673812365719742\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "\n",
    "# Create and train the model\n",
    "svm_model = SVC()\n",
    "svm_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Predict on test set\n",
    "y_pred = svm_model.predict(X_test_ss_scaled)\n",
    "\n",
    "# Calculate precision, recall, F1 score\n",
    "print(f\"Support Vector Machine - Precision: {precision_score(y_test, y_pred)}\")\n",
    "print(f\"Support Vector Machine - Recall: {recall_score(y_test, y_pred)}\")\n",
    "print(f\"Support Vector Machine - F1 Score: {f1_score(y_test, y_pred)}\")\n",
    "\n",
    "# Cross-validation scores\n",
    "cv_scores = cross_val_score(svm_model, X_train_ss_scaled, y_train, cv=5, scoring='accuracy')\n",
    "print(f\"Support Vector Machine - Cross-Validation Accuracy: {cv_scores.mean()}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression - Training Data Score: 0.8797968397291196\n",
      "Logistic Regression - Testing Data Score: 0.8963963963963963\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# Create and train the model\n",
    "logistic_regression_model = LogisticRegression()\n",
    "logistic_regression_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Score the model\n",
    "print(f\"Logistic Regression - Training Data Score: {logistic_regression_model.score(X_train_ss_scaled, y_train)}\")\n",
    "print(f\"Logistic Regression - Testing Data Score: {logistic_regression_model.score(X_test_ss_scaled, y_test)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression - Precision: 0.696969696969697\n",
      "Logistic Regression - Recall: 0.3898305084745763\n",
      "Logistic Regression - F1 Score: 0.5\n",
      "Logistic Regression - Cross-Validation Accuracy: 0.8713185326649162\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# Create and train the model\n",
    "logistic_regression_model = LogisticRegression()\n",
    "logistic_regression_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Predict on test set\n",
    "y_pred = logistic_regression_model.predict(X_test_ss_scaled)\n",
    "\n",
    "# Calculate precision, recall, F1 score\n",
    "print(f\"Logistic Regression - Precision: {precision_score(y_test, y_pred)}\")\n",
    "print(f\"Logistic Regression - Recall: {recall_score(y_test, y_pred)}\")\n",
    "print(f\"Logistic Regression - F1 Score: {f1_score(y_test, y_pred)}\")\n",
    "\n",
    "# Cross-validation scores\n",
    "cv_scores = cross_val_score(logistic_regression_model, X_train_ss_scaled, y_train, cv=5, scoring='accuracy')\n",
    "print(f\"Logistic Regression - Cross-Validation Accuracy: {cv_scores.mean()}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree - Training Data Score: 1.0\n",
      "Decision Tree - Testing Data Score: 0.8333333333333334\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "# Create and train the model\n",
    "decision_tree_model = DecisionTreeClassifier(random_state=1)\n",
    "decision_tree_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Score the model\n",
    "print(f\"Decision Tree - Training Data Score: {decision_tree_model.score(X_train_ss_scaled, y_train)}\")\n",
    "print(f\"Decision Tree - Testing Data Score: {decision_tree_model.score(X_test_ss_scaled, y_test)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree - Precision: 0.4\n",
      "Decision Tree - Recall: 0.5084745762711864\n",
      "Decision Tree - F1 Score: 0.44776119402985076\n",
      "Decision Tree - Cross-Validation Accuracy: 0.8250688310654889\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "# Create and train the model\n",
    "decision_tree_model = DecisionTreeClassifier(random_state=1)\n",
    "decision_tree_model.fit(X_train_ss_scaled, y_train)\n",
    "\n",
    "# Predict on test set\n",
    "y_pred = decision_tree_model.predict(X_test_ss_scaled)\n",
    "\n",
    "# Calculate precision, recall, F1 score\n",
    "print(f\"Decision Tree - Precision: {precision_score(y_test, y_pred)}\")\n",
    "print(f\"Decision Tree - Recall: {recall_score(y_test, y_pred)}\")\n",
    "print(f\"Decision Tree - F1 Score: {f1_score(y_test, y_pred)}\")\n",
    "\n",
    "# Cross-validation scores\n",
    "cv_scores = cross_val_score(decision_tree_model, X_train_ss_scaled, y_train, cv=5, scoring='accuracy')\n",
    "print(f\"Decision Tree - Cross-Validation Accuracy: {cv_scores.mean()}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dev",
   "language": "python",
   "name": "dev"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
