import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import holidays

# import darts
# import darts
# import matplotlib.pyplot as plt
# import darts
# from darts.models import RNNModel
# import seaborn
# import matplotlib
import holidays


def train_test_split_(X, y, train_until, validate_from):
    """
    Splits the overall dataset in train and test(validattion data)
    :param X: Independet variables (covariates)
    :param y: Dependent variable
    :return: Split of train and test data
    """
    #Train Test Split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 8, shuffle=False)
    #train_until = 285
    #validate_from = 286
    X_train = X.loc[:train_until]
    X_test = X.loc[validate_from:]
    y_train = y.loc[:train_until]
    y_test = y.loc[validate_from:]

    return X_train, X_test, y_train, y_test


def linear_model_(X_train, y_train, X_test):
    """
    Trains a linear Regression Model and predicts Values for the specified Timestamps.
    """
    # Initialize Model
    lr_model = linear_model.LinearRegression()

    # Train Linear Regression Model
    lr_model.fit(X_train, y_train)

    # Predict Datapoints
    y_pred = lr_model.predict(X_test)

    return y_pred


def support_vector_machines(X_train, y_train, X_test):
    """
    Support vector machines
    """
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def linear_model_lasso(X_train,y_train, X_test):
    """
    Trains a linear Regression Model and predicts Values for the specified Timestamps.
    """
    # Initialize Model
    #lr_model = linear_model.Lasso(alpha=0.1)
    lr_model = linear_model.Lasso(alpha=0.5)


    # Train Linear Regression Model
    lr_model.fit(X_train, y_train)

    # Predict Datapoints
    y_pred = lr_model.predict(X_test)

    return y_pred


def random_forest_regressor(X_train,y_train, X_test, max_depth, random_state):
    """
    Random Forest Regressor
    """
    regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    return y_pred

def neural_network(X_train, y_train, X_test, n_epochs):
    """
    Neural Network Keras from Tensorflow

    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize Model
    model = Sequential()
    model.add(Dense(4, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # Train Linear Regression Model
    model.fit(X_train, y_train, epochs=n_epochs)

    # Predict Datapoints
    y_pred_arr = model.predict(X_test)
    df_pred = pd.DataFrame(y_pred_arr, columns=['y_pred'])
    y_pred = df_pred['y_pred'].values
    return y_pred

def evaluate_predictions(y_test, y_pred):
    """
    Generate Evaluation Factors for the predictions
    """
    # RMSE = np.sqrt(((y_test - y_pred) ** 2).mean())
    # print("Root Mean Squared Error: ", RMSE)
    MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    # print("Mean Absolute Percentage Error: ", MAPE)
    # R2_score = r2_score(y_test, y_pred)
    return MAPE

def plot_results(df_pred, y_test,  plot_title, folder):
    """
    Plot results
    """

    # X_test = X_test[np.abs(X_test)< .1]= 0

    fig = plt.figure(figsize=[15, 7.5])
    #for y_pred, name, color in zip(lst_pred, lst_names, lst_colors):
    #    plt.plot(y_pred, color=color, label=name)
    plt.plot(df_pred["Lineare Regression"], color='green', label="Lineare Regression", linewidth = 0.75)
    plt.plot(df_pred["Random Forest Regressor"], color='red', label="Random Forest Regressor", linewidth = 0.75)
    plt.plot(df_pred["Lasso Regression"], color='blue', label="Lasso Regression", linewidth = 0.75)
    plt.plot(y_test, color='black', label="Actual Total Load")
    plt.xlabel('Validation Timeseries')
    plt.ylabel('Total Load')
    plt.title(plot_title)
    plt.grid()
    #plt.ylim([200, max(y_test) * 1.3])
    #plt.ylim([5000, 22000])
    plt.legend()
    # plt.show()
    #directory = os.path.join(
    #    r"C:\Users\pascs\OneDrive\Desktop\Master_Data_Science\Customer Data Analytics\CDA_Project\Prediction_Plots", folder)
    #if not os.path.exists(directory):
    #    os.makedirs(directory)
    fig.savefig(folder)

    return

def load_data(start, end,country_code, file_path, file_name, token):
    """
    Load Data from Transparency Platform
    """
    from entsoe import EntsoePandasClient
    import pandas as pd
    client = EntsoePandasClient(api_key=token)

    # print(client.query_load(country_code, start=start, end=end))
    print(client.query_load_and_forecast(country_code, start=start, end=end))

    df = client.query_load_and_forecast(country_code, start=start, end=end)

    df.to_csv(os.path.join(file_path, file_name), sep = ",")

    return df


def get_temporal_features_(index, features, n_cos_sin_hour=6, n_cos_sin_weekday=42, n_cos_sin_year=2):
    """
    get a df of temporal features specified for an index and the specified features
    """
    temporal_features = pd.DataFrame(index=index)
    if 'hour' in features:
        hour = index.hour
        for i in range(1, int(n_cos_sin_hour) + 1):
            hour_sin = np.sin(i * (2 / 24) * np.pi * hour).astype(float)
            hour_cos = np.cos(i * (2 / 24) * np.pi * hour).astype(float)
            temporal_features['hour_sin_{}'.format(i)] = hour_sin
            temporal_features['hour_cos_{}'.format(i)] = hour_cos

    if 'weekhour' in features:
        hour = index.hour + index.weekday * 24
        for i in range(1, int(n_cos_sin_weekday) + 1):
            hour_sin = np.sin(i * (2 / 24) * np.pi * hour).astype(float)
            hour_cos = np.cos(i * (2 / 24) * np.pi * hour).astype(float)
            temporal_features['weekhour_sin_{}'.format(i)] = hour_sin
            temporal_features['weekhour_cos_{}'.format(i)] = hour_cos

    if 'weekday' in features:
        weekday = index.weekday
        weekdays = pd.get_dummies(weekday, prefix='weekday').astype(float)
        weekdays.index = temporal_features.index
        temporal_features = pd.concat([temporal_features, weekdays], axis=1)

    if 'month' in features:
        month = index.month
        months = pd.get_dummies(month, prefix='month').astype(float)
        months.index = temporal_features.index
        temporal_features = pd.concat([temporal_features, months], axis=1)

    if 'dayofyear' in features:
        dayofyear = index.dayofyear
        for i in range(1, int(n_cos_sin_year) + 1):
            dayofyear_sin = np.sin(i * (2 / 365) * np.pi * dayofyear).astype(float)
            dayofyear_cos = np.cos(i * (2 / 365) * np.pi * dayofyear).astype(float)
            temporal_features['dayofyear_sin_{}'.format(i)] = dayofyear_sin
            temporal_features['dayofyear_cos_{}'.format(i)] = dayofyear_cos

    if 'holiday' in features:
        ch_holidays = holidays.CountryHoliday("CH")
        # ch_holidays = CountryHoliday('CH')
        # holiday = pd.Series(index.date).apply(lambda x: x in ch_holidays)
        # temporal_features['holiday'] = holiday.astype(int)
        temporal_features['holiday'] = pd.Series(index, index=index, name='holiday').apply(
            lambda x: x in ch_holidays).astype(int)
    if 'tz_shift' in features:
        """
        if there is a time shift (CET <-> CEST) then all the timestamp at this day = 1, no shift at this day = 0 
        """
        df_tz_info = pd.DataFrame()
        df_iteration = pd.DataFrame(data=[np.nan] * len(index), index=index).groupby(pd.Grouper(freq='D'))
        for key, daily_values in df_iteration:
            lst_tzinfo = [timestamp.tzinfo for timestamp in daily_values.index]
            if (all(tz_info == lst_tzinfo[0] for tz_info in lst_tzinfo)):
                df_tz_append = pd.DataFrame(data=[0] * len(daily_values), index=daily_values.index)
            else:
                df_tz_append = pd.DataFrame(data=[1] * len(daily_values), index=daily_values.index)
            df_tz_info = pd.concat([df_tz_info, df_tz_append], axis=0)
        df_tz_info.rename(columns={0: 'tz_shift'}, inplace=True)
        temporal_features['tz_shift'] = df_tz_info
    return temporal_features



if __name__ == "__main__":

    #Set Parameters for API-Query
    # from dotenv import load_dotenv
    # load_dotenv()
    # token = os.environ.get('API_KEY')
    token = "431ee029-249a-4cd3-a4f6-b3ea8043eb34"
    start = pd.Timestamp('20180101', tz='Europe/Zurich')
    end = pd.Timestamp('20211130', tz='Europe/Zurich')
    country_code = 'CH'
    file_path = "G:\Meine Ablage\Master_Data_Science\Energy Systems and IoT\Project"
    file_name = "Total_Load_Actual.csv"

    #Load Data from Transparency Platform and put it to CSV
    #df = load_data(start, end,country_code, file_path, file_name, token)

    #----------------------------------------------------------------------------------------------------------------

    #Get Data from CSV
    df = pd.read_csv(os.path.join(file_path, file_name), sep=',', index_col=0)

    #----------------------------------------------------------------------------------------------------------------
    #Create Covariates
    #tbd
    #Temperature, Wheather
    #Temporal Features

    #----------------------------------------------------------------------------------------------------------------

    #Select Timeseries as target variables and covariates
    df.index = pd.to_datetime(df.index, utc= True).tz_convert("CET")

    df_temp = pd.read_csv(os.path.join(file_path, "temperature.csv"), sep = ";", index_col=0)
    df_temp.index = pd.to_datetime(df_temp.index, utc=True).tz_convert("CET")

    #Get Intersection between temperature and actual load (some data is missing)
    lst_intersection = list(sorted(set(df.index.tolist()).intersection(df_temp.index.tolist()).intersection(df.index.tolist())))
    df_temp = df_temp.loc[lst_intersection]
    df = df.loc[df_temp.index]


    cov = get_temporal_features_(df.index, ["weekhour", "dayofyear","weekday", "hour", "holiday"], n_cos_sin_hour = 6, n_cos_sin_weekday = 42, n_cos_sin_year = 2)


    df = pd.concat([df,cov,df_temp], axis = 1)



    df = df.loc["2019-01-01 00:00:00+01:00":]
    covariates = [df[df.columns.tolist()[2:]],df[["OBFELDEN"]],df[df.columns.tolist()[2:-1]] ]

    covariate_names = ["Temporal Features + Temperature", "Temperature", "Temporal Features"]

    df_eval = pd.DataFrame()
    df_eval2 =pd.DataFrame()

    for X, name in zip(covariates, covariate_names):

        y = df["Actual Load"]

        # ----------------------------------------------------------------------------------------------------------------

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split_(X, y,train_until="2020-12-31 23:00:00+01:00", validate_from="2021-01-01 01:00:00+01:00")

        #-----------------------------------------------------------------------------------------------------------------
        #Train Models:

        # Linear Model
        y_pred_lin = linear_model_(X_train, y_train, X_test)

        #Random Forest Regressor
        y_pred_RFR = random_forest_regressor(X_train, y_train, X_test, max_depth=4, random_state=0)

        # Support Vector Machines
        #y_pred_SVM = support_vector_machines(X_train, y_train, X_test)

        # Lasso Regression
        y_pred_LAREG = linear_model_lasso(X_train, y_train, X_test)

        #Neural Network Tensorflow
        #y_pred_NN = neural_network(X_train, y_train, X_test, n_epochs = 800)

        lst_pred = [y_pred_lin, y_pred_RFR, y_pred_LAREG]
        lst_pred_names = ["Lineare Regression", "Random Forest Regressor", "Lasso Regression"]

        #------------------------------------------------------------------------------------------------------------------
        #Evaluate Predictions (MAPE)
        lst_MAPE = [evaluate_predictions(y_test, y_pred) for y_pred in lst_pred]
        df_MAPE = pd.DataFrame(data = lst_MAPE, index =lst_pred_names, columns=[name])
        df_eval = pd.concat([df_eval, df_MAPE], axis = 1)

        #-----------------------------------------------------------------------------------------------------------------
        #Plot Results
        df_pred = pd.DataFrame(data = lst_pred, index = lst_pred_names, columns = y_test.index).transpose()
        #lst_colors = ["red", "green", "blue", "orange", "yellow", "purple", "lime", "magenta"][:len(lst_pred_names)]

        plot_results(df_pred, y_test, plot_title = "Total Load Comparison Covariates: "+name, folder= os.path.join(file_path,name+".jpg"))

        lst_timehorizon = ["Day-Ahead", "Week-Ahead", "Month-Ahead", "6-Months Ahead"]
        lst_timestamp_until = ["2021-01-02 01:00:00+01:00", "2021-01-14 01:00:00+01:00", "2021-02-01 01:00:00+01:00", "2021-07-01 01:00:00+02:00"]

        df_results = pd.DataFrame()
        for timehorizon, lst_timestamp_until in zip (lst_timehorizon, lst_timestamp_until):

            MAPE = evaluate_predictions(y_test.loc[:lst_timestamp_until], pd.DataFrame(y_pred_lin, index =y_test.index).loc[:lst_timestamp_until][0])
            df_MAPE = pd.DataFrame(data=MAPE, index=[timehorizon], columns=[name])
            df_results = pd.concat([df_results, df_MAPE], axis = 0)
        df_eval2 = pd.concat([df_eval2, df_results], axis = 1)


        print(df_eval)
        print(df_eval2)
    df_eval.to_csv(os.path.join(file_path,"Evaluation_Different_Models.csv"))
    df_eval2.to_csv(os.path.join(file_path,"Evaluation_Different_Time_Horizons.csv"))





