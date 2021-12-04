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
from dotenv import load_dotenv
import holidays

# import darts
# import darts
# import matplotlib.pyplot as plt
# import darts
# from darts.models import RNNModel
# import seaborn
# import matplotlib
import holidays


def train_test_split_(X, y):
    """
    Splits the overall dataset in train and test(validattion data)
    :param X: Independet variables (covariates)
    :param y: Dependent variable
    :return: Split of train and test data
    """
    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 8, shuffle=False)
    #train_until = 285
    #validate_from = 286
    #X_train = X.loc[:train_until]
    #X_test = X.loc[validate_from:]
    #y_train = y.loc[:train_until]
    #y_test = y.loc[validate_from:]

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

def plot_results(lst_pred, lst_names, lst_colors,y_test,  plot_title, folder):
    """
    Plot results
    """

    # X_test = X_test[np.abs(X_test)< .1]= 0

    fig = plt.figure(figsize=[10, 7.5])
    for y_pred, name, color in zip(lst_pred, lst_names, lst_colors):
        plt.plot(y_pred, color=color, label=name)

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
    #fig.savefig(os.path.join(directory,plot_title.replace('/','')+'.jpeg'))

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

if __name__ == "__main__":

    #Set Parameters for API-Query
    load_dotenv()
    token = os.environ.get('API_KEY')
    start = pd.Timestamp('20211126', tz='Europe/Zurich')
    end = pd.Timestamp('20211127', tz='Europe/Zurich')
    country_code = 'CH'
    file_path = "G:\Meine Ablage\Master_Data_Science\Energy Systems and IoT\Project"
    file_name = "Total_Load_Actual.csv"

    #Load Data from Transparency Platform and put it to CSV
    #df = load_data(start, end,country_code, file_path, file_name)

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
    X = df[["Forecasted Load","Forecasted Load"]] # Forecasted Load was chosen as a dummy series for covariates, it should be replaced by Temperature, other Wheater Data and Temporal Features
    y = df["Actual Load"]

    # ----------------------------------------------------------------------------------------------------------------

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split_(X, y)

    #-----------------------------------------------------------------------------------------------------------------
    #Train Models:

    # Linear Model
    y_pred_lin = linear_model_(X_train, y_train, X_test)

    #Random Forest Regressor
    y_pred_RFR = random_forest_regressor(X_train, y_train, X_test, max_depth=4, random_state=0)

    # Support Vector Machines
    y_pred_SVM = support_vector_machines(X_train, y_train, X_test)

    # Lasso Regression
    y_pred_LAREG = linear_model_lasso(X_train, y_train, X_test)

    #Neural Network Tensorflow
    #y_pred_NN = neural_network(X_train, y_train, X_test, n_epochs = 800)

    lst_pred = [y_pred_lin, y_pred_RFR, y_pred_SVM, y_pred_LAREG]
    lst_pred_names = ["Lineare Regression", "Random Forest Regressor", "Support Vector Machines", "Lasso Regression"]

    #------------------------------------------------------------------------------------------------------------------
    #Evaluate Predictions (MAPE)

    lst_MAPE = [evaluate_predictions(y_test, y_pred) for y_pred in lst_pred]
    df_MAPE = pd.DataFrame(data = lst_MAPE, index =lst_pred_names, columns=["MAPE_Models"])
    print(df_MAPE)

    #-----------------------------------------------------------------------------------------------------------------
    #Plot Results
    lst_colors = ["red", "green", "blue", "orange", "yellow", "purple", "lime", "magenta"]
    plot_results(lst_pred, lst_pred_names, lst_colors, y_test, plot_title = "Total Load Comparison", folder= None)




