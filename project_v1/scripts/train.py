import pandas as pd
import numpy as np
import os
import joblib

from scripts.transform import TransformData

# ML models and utils
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


class TrainModel:
    transformed_df: pd.DataFrame  # input dataframe from the Transformed piece
    df_full: pd.DataFrame  # full dataframe with DUMMIES

    # Dataframes for ML
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    valid_df: pd.DataFrame
    train_valid_df: pd.DataFrame

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    X_train_valid: pd.DataFrame
    X_all: pd.DataFrame

    # feature sets
    GROWTH: list
    OHLCV: list
    CATEGORICAL: list
    TO_PREDICT: list
    TECHNICAL_INDICATORS: list
    TECHNICAL_PATTERNS: list
    MACRO: list
    NUMERICAL: list
    CUSTOM_NUMERICAL: list
    DUMMIES: list

    def __init__(self, transformed: TransformData):
        # init transformed_df
        self.transformed_df = transformed.transformed_df.copy(deep=True)
        self.transformed_df['ln_volume'] = self.transformed_df.Volume.apply(lambda x: np.log(x) if x > 0 else np.nan)
        # self.transformed_df['Date'] = pd.to_datetime(self.transformed_df['Date']).dt.strftime('%Y-%m-%d')

    def _define_feature_sets(self):
        self.GROWTH = [g for g in self.transformed_df if (g.find('growth_') == 0) & (g.find('future') < 0)]
        self.OHLCV = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']
        self.TO_PREDICT = [g for g in self.transformed_df.keys() if (g.find('future') >= 0)]
        self.MACRO = ['gdp_de_yoy', 'gdp_de_qoq', 'cpi_de_yoy', 'cpi_de_mom', 'bond10']
        self.CUSTOM_NUMERICAL = ['vix_adj_close', 'SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative',
                                 'volatility', 'ln_volume']

        # artifacts from joins and/or unused original vars
        self.TO_DROP = ['Year', 'Date', 'index', 'Quarter', 'index_x', 'index_y'] + self.CATEGORICAL + self.OHLCV

        # All Supported Ta-lib indicators: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/funcs.md
        self.TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1', 'aroon_2', 'aroonosc',
                                     'bop', 'cci', 'cmo', 'dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
                                     'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
                                     'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
                                     'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
                                     'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
                                     'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
                                     'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
                                     'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']
        self.TECHNICAL_PATTERNS = [g for g in self.transformed_df.keys() if g.find('cdl') >= 0]

        self.NUMERICAL = self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + \
                         self.CUSTOM_NUMERICAL + self.MACRO

        # CHECK: NO OTHER INDICATORS LEFT
        self.OTHER = [k for k in self.transformed_df.keys() if
                      k not in self.OHLCV + self.CATEGORICAL + self.NUMERICAL + self.TO_DROP + self.TO_PREDICT]
        return

    def _define_dummies(self):
        # dummy variables can't be generated from Date and numeric variables ==> convert to STRING (to define groups for Dummies)
        # self.transformed_df.loc[:,'Month'] = self.transformed_df.Month_x.dt.strftime('%B')
        self.transformed_df.loc[:, 'Month'] = self.transformed_df.Month.astype(str)
        self.transformed_df['Weekday'] = self.transformed_df['Weekday'].astype(str)

        # Generate dummy variables (no need for bool, let's have int32 instead)
        dummy_variables = pd.get_dummies(self.transformed_df[self.CATEGORICAL], dtype='int32')
        self.df_full = pd.concat([self.transformed_df, dummy_variables], axis=1)
        # get dummies names in a list
        self.DUMMIES = dummy_variables.keys().to_list()

    def _perform_temporal_split(self, df: pd.DataFrame, min_date, max_date, train_prop=0.8, val_prop=0.1,
                                test_prop=0.1):
        """
    Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.

    Args:
        df (DataFrame): The DataFrame to split.
        min_date (str or Timestamp): Minimum date in the DataFrame.
        max_date (str or Timestamp): Maximum date in the DataFrame.
        train_prop (float): Proportion of data for training set (default: 0.7).
        val_prop (float): Proportion of data for validation set (default: 0.15).
        test_prop (float): Proportion of data for test set (default: 0.15).

    Returns:
        DataFrame: The input DataFrame with a new column 'split' indicating the split for each row.
    """
        # Define the date intervals
        train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
        val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

        # Assign split labels based on date ranges
        split_labels = []
        for date in df['Date']:
            if date <= train_end:
                split_labels.append('train')
            elif date <= val_end:
                split_labels.append('validation')
            else:
                split_labels.append('test')

        # Add 'split' column to the DataFrame
        df['split'] = split_labels

        return df

    def _define_dataframes_for_ML(self):

        features_list = self.NUMERICAL + self.DUMMIES
        # What we're trying to predict?
        to_predict = 'is_positive_growth_3d_future'

        self.train_df = self.df_full[self.df_full.split.isin(['train'])].copy(deep=True)
        self.valid_df = self.df_full[self.df_full.split.isin(['validation'])].copy(deep=True)
        self.train_valid_df = self.df_full[self.df_full.split.isin(['train', 'validation'])].copy(deep=True)
        self.test_df = self.df_full[self.df_full.split.isin(['test'])].copy(deep=True)

        # Separate numerical features and target variable for training and testing sets
        self.X_train = self.train_df[features_list + [to_predict]]
        self.X_valid = self.valid_df[features_list + [to_predict]]
        self.X_train_valid = self.train_valid_df[features_list + [to_predict]]
        self.X_test = self.test_df[features_list + [to_predict]]
        # this to be used for predictions and join to the original dataframe new_df
        self.X_all = self.df_full[features_list + [to_predict]].copy(deep=True)

        # Clean from +-inf and NaNs:

        self.X_train = self._clean_dataframe_from_inf_and_nan(self.X_train)
        self.X_valid = self._clean_dataframe_from_inf_and_nan(self.X_valid)
        self.X_train_valid = self._clean_dataframe_from_inf_and_nan(self.X_train_valid)
        self.X_test = self._clean_dataframe_from_inf_and_nan(self.X_test)
        self.X_all = self._clean_dataframe_from_inf_and_nan(self.X_all)

        self.y_train = self.X_train[to_predict]
        self.y_valid = self.X_valid[to_predict]
        self.y_train_valid = self.X_train_valid[to_predict]
        self.y_test = self.X_test[to_predict]
        self.y_all = self.X_all[to_predict]

        # remove y_train, y_test from X_ dataframes
        del self.X_train[to_predict]
        del self.X_valid[to_predict]
        del self.X_train_valid[to_predict]
        del self.X_test[to_predict]
        del self.X_all[to_predict]

        print(f'length: X_train {self.X_train.shape},  X_validation {self.X_valid.shape}, X_test {self.X_test.shape}')
        print(f'  X_train_valid = {self.X_train_valid.shape},  all combined: X_all {self.X_all.shape}')

    def _clean_dataframe_from_inf_and_nan(self, df: pd.DataFrame):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def prepare_dataframe(self):
        print("Prepare the dataframe: define feature sets, add dummies, temporal split")
        self._define_feature_sets()
        # get dummies and df_full
        self._define_dummies()

        # temporal split
        min_date_df = self.df_full.Date.min()
        max_date_df = self.df_full.Date.max()
        self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)

        # define dataframes for ML
        self._define_dataframes_for_ML()

        return

    def train_random_forest(self, max_depth=17, n_estimators=200):
        # https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
        print('Training the best model (RandomForest (max_depth=17, n_estimators=200))')
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=42,
                                            n_jobs=-1)

        self.model = self.model.fit(self.X_train_valid, self.y_train_valid)

    def persist(self, data_dir: str):
        '''Save dataframes to files in a local directory 'dir' '''
        os.makedirs(data_dir, exist_ok=True)

        # Save the model to a file
        model_filename = 'random_forest_model.joblib'
        path = os.path.join(data_dir, model_filename)
        joblib.dump(self.model, path)

    def load(self, data_dir: str):
        """Load files from the local directory"""
        os.makedirs(data_dir, exist_ok=True)
        # Save the model to a file
        model_filename = 'random_forest_model.joblib'
        path = os.path.join(data_dir, model_filename)

        self.model = joblib.load(path)

    def make_inference(self, pred_name: str):
        # https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
        print('Making inference and calculating financial results')

        y_pred_all = self.model.predict_proba(self.X_all)
        y_pred_all_class1 = [k[1] for k in y_pred_all]  # list of predictions for class "1"
        y_pred_all_class1_array = np.array(y_pred_all_class1)  # (Numpy Array) np.array of predictions for class "1" , converted from a list

        self.df_full[pred_name] = y_pred_all_class1_array

        y_pred_all1 = self.model.predict(self.X_all)
        y_pred_all1_class1 = [k for k in y_pred_all1]  # list of predictions for class "1"
        y_pred_all1_class1_array = np.array(y_pred_all1_class1)  # (Numpy Array) np.array of predictions for class "1" , converted from a list

        self.df_full[pred_name + "_notProba"] = y_pred_all1_class1_array

        # define rank of the prediction
        self.df_full[f"{pred_name}_rank"] = self.df_full.groupby("Date")[pred_name].rank(method="first",
                                                                                         ascending=False)

        def get_predictions_correctness(df: pd.DataFrame, to_predict: str, predictions = str(pred_name + "_notProba")):
            PREDICTIONS = [predictions]
            print(f'Prediction columns founded: {PREDICTIONS}')

            # add columns is_correct_
            for pred in PREDICTIONS:
                part1 = pred.split('_')[0]  # first prefix before '_'
                df[f'is_correct_{part1}'] = (df[pred] == df[to_predict]).astype(int)

            # IS_CORRECT features set
            IS_CORRECT = [k for k in df.keys() if k.startswith('is_correct_')]
            print(f'Created columns is_correct: {IS_CORRECT}')

            print('Precision on TEST set for each prediction:')
            # define "Precision" for ALL predictions on a Test dataset (~4 last years of trading)
            for i, column in enumerate(IS_CORRECT):
                prediction_column = PREDICTIONS[i]
                is_correct_column = column
                filter = (df.split == 'test') & (df[prediction_column] == 1)
                print(f'Prediction column:{prediction_column} , is_correct_column: {is_correct_column}')
                print(df[filter][is_correct_column].value_counts())
                print(df[filter][is_correct_column].value_counts() / len(df[filter]))
                print('---------')

            return PREDICTIONS, IS_CORRECT

        to_predict = 'is_positive_growth_3d_future'
        PREDICTIONS, IS_CORRECT = get_predictions_correctness(df=self.df_full, to_predict=to_predict)

        # Calculate fin. result for ALL predictions (manual and produced by models)
        # Strategy is 50$ investment for each positive prediction (1) for 3 days ahead

        investment = 50  # variable for the investement value
        yTrad = round(self.df_full[self.df_full.split == 'test'].Date.nunique() / 252, 1)  # variable for the number of years of trading

        sim1_results = []  # results in Array

        # Iterate over all predictions
        for pred in PREDICTIONS:
            print(f'Calculating sumulation for prediction {pred}:')
            print(f"    Count times of investment {len(self.df_full[(self.df_full.split == 'test') & (self.df_full[pred] == 1)])} out of {len(self.df_full[(self.df_full.split == 'test')])} TEST records")

            # Prefix: e.g. pred1 or pred10
            pred_prefix = pred.split('_')[0]

            # Fin. result columns: define new records for EACH positive prediction
            self.df_full['sim1_gross_rev_' + pred_prefix] = self.df_full[pred] * investment * (self.df_full['growth_future_3d'] - 1)
            self.df_full['sim1_fees_' + pred_prefix] = -self.df_full[pred] * investment * 0.002
            self.df_full['sim1_net_rev_' + pred_prefix] = self.df_full['sim1_gross_rev_' + pred_prefix] + self.df_full[
                'sim1_fees_' + pred_prefix]

            # calculate agg. results for each PREDICTION columns (pred) on TEST
            filter_test_and_positive_pred = (self.df_full.split == 'test') & (self.df_full[
                                                                            pred] == 1)  # filter records on TEST set, when current prediction is 1 (we invest $50 for 3 days ahead - 3 periods)
            sim1_count_investments = len(self.df_full[filter_test_and_positive_pred])
            sim1_gross_rev = self.df_full[filter_test_and_positive_pred]['sim1_gross_rev_' + pred_prefix].sum()
            sim1_fees = self.df_full[filter_test_and_positive_pred]['sim1_fees_' + pred_prefix].sum()
            sim1_net_rev = self.df_full[filter_test_and_positive_pred]['sim1_net_rev_' + pred_prefix].sum()

            if sim1_gross_rev > 0:
                sim1_fees_percentage = -sim1_fees / sim1_gross_rev
            else:
                sim1_fees_percentage = None

            if sim1_count_investments > 0:
                sim1_average_net_revenue = sim1_net_rev / sim1_count_investments
            else:
                sim1_average_net_revenue = None

            # APPROXIMATE CAPITAL REQUIRED and CAGR Calculation
            df_investments_count_daily = pd.DataFrame(
                self.df_full[filter_test_and_positive_pred].groupby('Date')[pred].count())
            sim1_avg_investments_per_day = df_investments_count_daily[pred].mean()
            sim1_q75_investments_per_day = df_investments_count_daily[pred].quantile(
                0.75)  # 75% case - how many $50 investments per day do we have?
            # df_investments_count_daily[pred].mean()
            sim1_capital = investment * 3 * sim1_q75_investments_per_day  # 3 days in a row with positive predictions
            # CAGR: average growth per year. E.g. if you have 1.5 return (50% growth in 4 years) --> (1.5)**(1/4) = 1.106 or 10.6% average
            sim1_CAGR = ((sim1_capital + sim1_net_rev) / sim1_capital) ** (1 / yTrad)

            # append to DF
            sim1_results.append(
                (pred, sim1_count_investments, sim1_gross_rev, sim1_fees, sim1_net_rev, sim1_fees_percentage,
                 sim1_average_net_revenue, sim1_avg_investments_per_day, sim1_capital, sim1_CAGR))

            # output for all predictions with some positive predictions
            if sim1_count_investments > 1:
                print(
                    f"    Financial Result: \n {self.df_full[filter_test_and_positive_pred][['sim1_gross_rev_' + pred_prefix, 'sim1_fees_' + pred_prefix, 'sim1_net_rev_' + pred_prefix]].sum()}")
                print(f"        Count Investments in {yTrad} years (on TEST): {sim1_count_investments}")
                print(f"        Gross Revenue: ${int(sim1_gross_rev)}")
                print(f"        Fees (0.2% for buy+sell): ${int(-sim1_fees)}")
                print(f"        Net Revenue: ${int(sim1_net_rev)}")
                print(f"        Fees are {int(-investment * sim1_fees / sim1_gross_rev)} % from Gross Revenue")
                print(f"        Capital Required : ${int(sim1_capital)} (Vbegin)")
                print(f"        Final value (Vbegin + Net_revenue) : ${int(sim1_capital + sim1_net_rev)} (Vfinal)")

                print(
                    f"        Average CAGR on TEST ({yTrad} years) : {np.round(sim1_CAGR, 3)}, or {np.round(100.0 * (sim1_CAGR - 1), 1)}% ")

                print(f"        Average daily stats: ")
                print(
                    f"            Average net revenue per investment: ${np.round(sim1_net_rev / sim1_count_investments, 2)} ")
                print(f"            Average investments per day: {int(np.round(sim1_avg_investments_per_day))} ")
                print(f"            Q75 investments per day: {int(np.round(sim1_q75_investments_per_day))} ")
                print('=============================================+')

        # results in a DataFrame from an Array
        #columns_simulation = ['prediction', 'sim1_count_investments', 'sim1_gross_rev', 'sim1_fees', 'sim1_net_rev',
        #                      'sim1_fees_percentage', 'sim1_average_net_revenue', 'sim1_avg_investments_per_day',
        #                      'sim1_capital', 'sim1_CAGR']

        #df_sim1_results = pd.DataFrame(sim1_results, columns=columns_simulation)


