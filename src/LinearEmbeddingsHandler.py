import pandas as pd
import pickle
import os
import sys
import torch
from .EmbeddingsPreprocess import EmbeddingsPreprocess
from .mcrmse_score import mcrmse_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


class LinearEmbeddingsHandler():
    def __init__(self, cnf, data = None):
        self.data = data
        self.cnf = cnf
        if os.path.isfile(self.cnf.files.embeddings_data):
            self.embeds_data = pd.read_csv(self.cnf.files.embeddings_data)
        else:
            self.embeds_data = None
        self.TARGET_COLS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        self.model_dict = {}


    def preprocess(self):
        print("\n__PREPROCESSING__\n")
        if self.data is not None:
            embeds_obj = EmbeddingsPreprocess(self.data, self.cnf)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            embeds_obj.create_embeddings(device)
            self.embeds_data = self.data[self.TARGET_COLS].join(embeds_obj.embeddings_data)
            self.embeds_data.to_csv(self.cnf.files.embeddings_data, index=False)
        else:
            sys.exit(f'No {self.cnf.files.train_data} file available, cannot run preprocessing!')
    

    def grid_search(self):
        print("\n__GRID SEARCH__\n")
        param_grid = {'C': self.cnf.embeddings_models.param_grid.C, 'degree': self.cnf.embeddings_models.param_grid.degree, 
                          'gamma': self.cnf.embeddings_models.param_grid.gamma, 'kernel': self.cnf.embeddings_models.param_grid.kernel,
                          'epsilon': self.cnf.embeddings_models.param_grid.epsilon }
                          
        for col in self.TARGET_COLS:
            grid = GridSearchCV(estimator = SVR(), param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error')
            grid.fit(self.X_train, self.y_train[col])

            print(f"For column '{col}' best parameters are: {grid.best_params_}")
            self.model_dict[col] = (grid.best_estimator_)


    def train(self):
        if self.embeds_data is None:
            print('\n__NO EMBEDDINGS DATA AVAILABLE, PREPROCESSING WILL RUN FIRST__\n')
            self.preprocess()

        self.X = self.embeds_data.drop(columns=self.TARGET_COLS, axis=1)
        self.y = self.embeds_data[self.TARGET_COLS]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=123, test_size=0.2)

        print("\n__TRAINING__\n")
        if self.cnf.embeddings_models.use_gridsearch == True:
            self.grid_search()
        else:
            self.model_dict = {'cohesion': SVR(C=self.cnf.embeddings_models.SVR_cohesion.C, epsilon=self.cnf.embeddings_models.SVR_cohesion.epsilon, gamma=self.cnf.embeddings_models.SVR_cohesion.gamma, kernel=self.cnf.embeddings_models.SVR_cohesion.kernel),
                               'syntax': SVR(C=self.cnf.embeddings_models.SVR_syntax.C, epsilon=self.cnf.embeddings_models.SVR_syntax.epsilon, gamma=self.cnf.embeddings_models.SVR_syntax.gamma, kernel=self.cnf.embeddings_models.SVR_syntax.kernel),
                               'vocabulary': SVR(C=self.cnf.embeddings_models.SVR_vocabulary.C, epsilon=self.cnf.embeddings_models.SVR_vocabulary.epsilon, gamma=self.cnf.embeddings_models.SVR_vocabulary.gamma, kernel=self.cnf.embeddings_models.SVR_vocabulary.kernel),
                               'phraseology': SVR(C=self.cnf.embeddings_models.SVR_phraseology.C, epsilon=self.cnf.embeddings_models.SVR_phraseology.epsilon, gamma=self.cnf.embeddings_models.SVR_phraseology.gamma, kernel=self.cnf.embeddings_models.SVR_phraseology.kernel),
                               'grammar': SVR(C=self.cnf.embeddings_models.SVR_grammar.C, epsilon=self.cnf.embeddings_models.SVR_grammar.epsilon, gamma=self.cnf.embeddings_models.SVR_grammar.gamma, kernel=self.cnf.embeddings_models.SVR_grammar.kernel),
                               'conventions': SVR(C=self.cnf.embeddings_models.SVR_conventions.C, epsilon=self.cnf.embeddings_models.SVR_conventions.epsilon, gamma=self.cnf.embeddings_models.SVR_conventions.gamma, kernel=self.cnf.embeddings_models.SVR_conventions.kernel)}

        y_pred = []
        for column in self.TARGET_COLS:
            self.model_dict[column].fit(self.X_train.values, self.y_train[column].values)
            y_pred.append(self.model_dict[column].predict(self.X_test.values))
        
        mcrmse = mcrmse_score(self.y_test, y_pred)
        print('MCRMSE score: {}\n'.format(mcrmse))


    def predict(self, new_data):

        if new_data is not None:
            
            new_data_ids = new_data['text_id']

            if len(self.model_dict) == 0:
                print("\n__NO TRAINED MODEL AVAILABLE, TRAINING WILL RUN FIRST__\n")
                self.train()

            if os.path.isfile(self.cnf.files.embeddings_data_test) == False:
                print("\n__PREPROCESSING_TEST_DATA__\n")
                embeds_obj = EmbeddingsPreprocess(new_data, self.cnf)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                embeds_obj.create_embeddings(device)
                test_data = embeds_obj.embeddings_data
                test_data.to_csv(self.cnf.files.embeddings_data_test, index=False)
            else:
                test_data = pd.read_csv(self.cnf.files.embeddings_data_test)

            print("\n__PREDICTING__\n")
            y_pred = []
            for column in self.model_dict:
                y_pred.append(self.model_dict[column].predict(test_data.values))

            self.predictions_df = pd.DataFrame(scores for scores in y_pred).T
            self.predictions_df.columns = self.TARGET_COLS
            self.predictions_df = pd.concat([new_data_ids, self.predictions_df], axis=1)
            self.predictions_df.to_csv(self.cnf.files.embeddings_predictions, index=False)
            print("Predictions:")
            print(self.predictions_df.head())
            print(f'Full predictions dataframe saved to {self.cnf.files.embeddings_predictions}\n')
        else:
            sys.exit(f'No {self.cnf.files.test_data} file available, cannot get predictions!')


    def load_models(self):
        print("\n__LOADING_MODELS__\n")
        for column in self.TARGET_COLS:
            mod_file_name = str(column) + '_embeddings_model.pickle'
            mod_full_path = os.path.join(self.cnf.embeddings_models.model_path, mod_file_name)
            if os.path.isfile(mod_full_path):
                with open(mod_full_path, 'rb') as mod:
                    self.model_dict[column] = pickle.load(mod)
            else:
                print(f"__NO SAVED MODEL AVAILABLE FOR COLUMN {column}__")


    def save_models(self):
        print("\n__SAVING_MODELS__\n")
        if len(self.model_dict) == 0:
            print("__NO TRAINED MODEL AVAILABLE, TRAINING WILL RUN FIRST__\n")
            self.train()
            print("__SAVING__\n")

        for column in self.TARGET_COLS:
            mod_file_name = str(column)+'_embeddings_model.pickle'
            mod_full_path = os.path.join(self.cnf.embeddings_models.model_path, mod_file_name)
            if os.path.isfile(mod_full_path):
                print(f'Previous {column} model will be overwritten.')
            with open(mod_full_path, 'wb') as f:
                pickle.dump(self.model_dict[column], f)