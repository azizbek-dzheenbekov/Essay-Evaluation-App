import pandas as pd
import os

class Orchestrator():
    def __init__(self, cnf, modelHandler):
        self.cnf = cnf
        if os.path.isfile(self.cnf.files.train_data):
            self.train_data = pd.read_csv(cnf.files.train_data)
            self.modelHandler = modelHandler(cnf, self.train_data)
        else:
            self.modelHandler = modelHandler(cnf)
        if os.path.isfile(self.cnf.files.test_data):
            self.test_data = pd.read_csv(cnf.files.test_data)
        else: 
            self.test_data = None
        

    def orchestrate(self):
        if self.cnf.main.preprocess == True:
            self.preprocess()
        if self.cnf.main.train == True:
            self.train()
        if self.cnf.main.load == True:
            self.load_models()
        if self.cnf.main.save == True:
            self.save_models()
        if self.cnf.main.predict == True:
            self.predict()

    def preprocess(self):
        self.modelHandler.preprocess()

    def train(self):
        self.modelHandler.train()

    def predict(self):
        self.modelHandler.predict(self.test_data)

    def load_models(self):
        self.modelHandler.load_models()

    def save_models(self):
        self.modelHandler.save_models()