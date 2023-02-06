import pandas as pd
import pickle
import os
import sys

from .FeaturePreprocess import FeaturePreprocess
from .mcrmse_score import mcrmse_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

class LinearFeaturesHandler():
    def __init__(self, cnf, data = None):
        self.data = data
        self.cnf = cnf
        if os.path.isfile(self.cnf.files.features_data):
            self.features_data = pd.read_csv(self.cnf.files.features_data)
        else:
            self.features_data = None
        self.TARGET_COLS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.model_dict = {}

    def preprocess(self):
        print("\n__PREPROCESSING__\n")
        if self.data is not None:
            features_obj = FeaturePreprocess(self.data, self.cnf)
            features_obj.create_features()
            self.features_data = features_obj.data
            self.features_data.to_csv(self.cnf.files.features_data, index=False)
        else:
            sys.exit(f'No {self.cnf.files.train_data} file available, cannot run preprocessing!')

    def feature_scaling_for_train(self):
        self.scaler_X = StandardScaler()
        self.scaler_X.fit(self.X_train)

        X_train_scaled = self.scaler_X.transform(self.X_train)
        X_test_scaled = self.scaler_X.transform(self.X_test)

        self.X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.X.columns)
        self.X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.X.columns)

    def feature_scaling_for_predict(self, new_data):
        new_data = new_data.drop(columns=['text_id', 'full_text', 'cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'], axis=1, errors='ignore')
        self.new_X_scaled = pd.DataFrame(self.scaler_X.transform(new_data), columns=new_data.columns)

    def rmse_score(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

    def grid_search(self):
        print("\n__GRID SEARCH__\n")

        cv = RepeatedKFold(n_splits=5, n_repeats=5)
        rmse_scoring = make_scorer(self.rmse_score, greater_is_better = False)

        ElasticNet_param_grid = {'alpha': self.cnf.feature_models.ElasticNet_param_grid.alpha, 'l1_ratio': self.cnf.feature_models.ElasticNet_param_grid.l1_ratio}
        SVR_param_grid = {'C': self.cnf.feature_models.SVR_param_grid.C, 'epsilon': self.cnf.feature_models.SVR_param_grid.epsilon}

        elnet_grid = GridSearchCV(ElasticNet(), param_grid=ElasticNet_param_grid, cv=cv, scoring=rmse_scoring)
        elnet_grid.fit(self.X_train_scaled, self.y_train.iloc[:, 0])
        print(f"For column '{self.y_train.columns[0]}', best params: {elnet_grid.best_params_}")
        self.model_dict[self.y_train.columns[0]].append(elnet_grid.best_estimator_)

        for col in self.y_train.columns[1:]:
            scr_grid = GridSearchCV(SVR(kernel='linear'), param_grid=SVR_param_grid, cv=cv, scoring=rmse_scoring)
            scr_grid.fit(self.X_train_scaled, self.y_train[col])
            print(f"For column '{col}', best params: {scr_grid.best_params_}")
            self.model_dict[col].append(scr_grid.best_estimator_)

    def train(self):
        if self.features_data is None:
            print('\n__NO FEATURES DATA AVAILABLE, PREPROCESSING WILL RUN FIRST__\n')
            self.preprocess()

        self.X = self.features_data.drop(columns=['text_id', 'full_text', 'cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions'], axis=1)
        self.y = self.features_data[self.TARGET_COLS]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=123, test_size=0.2)

        self.feature_scaling_for_train()

        print("\n__TRAINING__\n")
        if self.cnf.feature_models.use_gridsearch == True:
            self.grid_search()
        else:
            self.model_dict = {'cohesion': ElasticNet(alpha=self.cnf.feature_models.ElasticNet_cohesion.alpha, l1_ratio=self.cnf.feature_models.ElasticNet_cohesion.l1_ratio),
                                'syntax': SVR(C=self.cnf.feature_models.SVR_syntax.C, epsilon=self.cnf.feature_models.SVR_syntax.epsilon, kernel=self.cnf.feature_models.SVR_syntax.kernel),
                                'vocabulary': SVR(C=self.cnf.feature_models.SVR_vocabulary.C, epsilon=self.cnf.feature_models.SVR_vocabulary.epsilon, kernel=self.cnf.feature_models.SVR_vocabulary.kernel),
                                'phraseology': SVR(C=self.cnf.feature_models.SVR_phraseology.C, epsilon=self.cnf.feature_models.SVR_phraseology.epsilon, kernel=self.cnf.feature_models.SVR_phraseology.kernel),
                                'grammar': SVR(C=self.cnf.feature_models.SVR_grammar.C, epsilon=self.cnf.feature_models.SVR_grammar.epsilon, kernel=self.cnf.feature_models.SVR_grammar.kernel),
                                'conventions': SVR(C=self.cnf.feature_models.SVR_conventions.C, epsilon=self.cnf.feature_models.SVR_conventions.epsilon, kernel=self.cnf.feature_models.SVR_conventions.kernel)}
        
        y_pred = []

        for column in self.TARGET_COLS:
            self.model_dict[column].fit(self.X_train_scaled, self.y_train[column])
            y_pred.append(self.model_dict[column].predict(self.X_test_scaled))
        
        mcrmse = mcrmse_score(self.y_test, y_pred)
        print('MCRMSE score: {}\n'.format(mcrmse))

    def predict(self, new_data):

        if new_data is not None:

            new_data_ids = new_data['text_id']

            if len(self.model_dict) == 0:
                print("\n__NO TRAINED MODEL AVAILABLE, TRAINING WILL RUN FIRST__\n")
                self.train()
            print("\n__PREPROCESSING_NEW_DATA__\n")
            features_obj = FeaturePreprocess(new_data, self.cnf)
            features_obj.create_features()
            new_data = features_obj.data

            self.feature_scaling_for_predict(new_data)

            print("\n__PREDICTING__\n")
            y_pred = []
            for column in self.model_dict:
                y_pred.append(self.model_dict[column].predict(self.new_X_scaled))

            self.predictions_df = pd.DataFrame(scores for scores in y_pred).T
            self.predictions_df.columns = self.TARGET_COLS
            self.predictions_df = pd.concat([new_data_ids, self.predictions_df], axis=1)
            self.predictions_df.to_csv(self.cnf.files.features_predictions, index=False)
            
            print("Predictions:")
            print(self.predictions_df.head())
            print(f'Full predictions dataframe saved to {self.cnf.files.features_predictions}\n')
        else:
            sys.exit(f'No {self.cnf.files.test_data} file available, cannot get predictions!')

    def load_models(self):
        print("\n__LOADING_MODELS__\n")
        self.model_dict = {}
        for column in self.TARGET_COLS:
            mod_file_name = str(column) + '_feature_model.pickle'
            mod_full_path = os.path.join(self.cnf.feature_models.model_path, mod_file_name)
            if os.path.isfile(mod_full_path):
                with open(mod_full_path, 'rb') as mod:
                    self.model_dict[column] = pickle.load(mod)
            else: 
                print(f"__NO SAVED MODEL AVAILABLE FOR COLUMN {column}__")
        scaler_file_name = 'std_scaler.pickle'
        scaler_full_path = os.path.join(self.cnf.feature_models.model_path, scaler_file_name)
        if os.path.isfile(scaler_full_path):
            with open(scaler_full_path, 'rb') as sclr:
                self.scaler_X = pickle.load(sclr)
        else: 
            print("__NO SAVED STANDARD SCALER AVAILABLE__")

    def save_models(self):
        print("\n__SAVING_MODELS__\n")
        if len(self.model_dict) == 0:
            print("__NO TRAINED MODEL AVAILABLE, TRAINING WILL RUN FIRST__\n")
            self.train()
            print("__SAVING__\n")
        for column in self.TARGET_COLS:
            mod_file_name = str(column)+'_feature_model.pickle'
            mod_full_path = os.path.join(self.cnf.feature_models.model_path, mod_file_name)
            if os.path.isfile(mod_full_path):
                print(f'Previous {column} model will be overwritten.')
            with open(mod_full_path, 'wb') as f:
                pickle.dump(self.model_dict[column], f)
        scaler_file_name = 'std_scaler.pickle'
        scaler_full_path = os.path.join(self.cnf.feature_models.model_path, scaler_file_name)
        if os.path.isfile(scaler_full_path):
            print(f'Previous StandardScaler file will be overwritten.')
        with open(scaler_full_path, 'wb') as f:
                pickle.dump(self.scaler_X, f)
        print()