# Essay Evaluation Tool
Welcome to the Essay Evaluation tool! This tool allows you to preprocess essays, train models, and predict six essay evaluation scores (cohesion, syntax, vocabulary, phraseology, grammar and conventions) using either **feature engineering combined with linear models** (ElasticNet and Support Vector Regression (SVR)) or  **DeBERTa/BERT embeddings combined with SVR**.
## Features Linear Model
The first model is a feature engineering model that uses ElasticNet and SVR to predict different essay score values. The model is trained on a dataset of essays and their corresponding scores. To make predictions it uses 45 features of the essays, such as average sentence count per paragraph, spelling mistakes percent, unique linking words count, number of phrasal verbs, part-of-speech tags and many more.
## Embeddings Model
The second model uses the pretrained deBERTa (or BERT) model to generate embeddings from the text. These embeddings are then used as input to a Support Vector Regression (SVR) model, which is trained to predict the essay scores. This model is trained on a dataset of essays and their corresponding scores, and uses the generated embeddings as input features to make predictions. WARNING: preprocessing data without GPU takes around 25 hours for DeBERTa and 2 hours for BERT model.
## How to use
### Dependencies
This application requires the following libraries (they are also stored in *requirements.txt* file):
- hydra-core==1.3.1
- lexicalrichness==0.3.1
- nltk==3.8
- numpy==1.24.1
- pandas==1.5.2
- pyenchant==3.2.2
- scikit_learn==1.2.0
- torch==1.13.1
- tqdm==4.64.1
- transformers==4.25.1

To install all the needed libraries use the following command while being in the project directory:
```python
pip install -r requirements.txt
```

### Configurations
Before running the application, make sure to set configuration options in *config.yaml* file to get the desired results. The available options are:
- **main:**  
This section specifies the main configurations of the application. This section should be looked through each time and set accordingly to users needs.
    - **modelHandler:** Specifies the model handler that will be used. Options are **_LinearEmbeddingsHandler_** (to use deBERTa (or BERT) model embeddings) or **_LinearFeaturesHandler_** (to use engineered features);
    - **preprocess:** Set to **_True_** to preprocess the data or **_False_** to use the previously saved preprocessed data. If no previously saved preprocessed data available, prepocessing will run even if set to **_False_**;
    - **train:** Set to **_True_** to train the model, or **_False_** to skip model training;
    - **load:** Set to **_True_** to load the previously saved trained models, or **_False_** to skip this step. If no previously saved models available, training will run even if set to **_False_**;
    - **save:** Set to **_True_** to save the trained model, or **_False_** to skip this step;
    - **predict:** Set to **_True_** to make predictions on the new data using the trained model, or **_False_** to skip this step.
- **files:**  
This section specifies the file paths for the various data and text files used by the application. It is required to have a *train.csv* file to run training, and *test.csv* file to run predicting. Files *features_data.csv* and *embeddings_data.csv* store preprocessed data. They are optional and generated automatically after running preprocessing.Text files *phrasal_verbs.txt*,  *connectives.txt*, *pos_list.txt*, *two_word_phrases.txt* and *three_word_phrases.txt* are also required for feature engineering and can be edited to suit the users needs. Files *feature_predictions.csv* and *embeddings_predictions.csv* store all predictions and are generaded automatically after running predicting.
- **feature_models:**  
This section specifies the default model hyperparameters for the linear features models, an option to run **GridSearchCV** with prefered parameter grid and the directory where the model will be saved.
- **embeddings_params:**  
This section specifies parameters for the embeddings extraction process. It is possible to specify the **MODEL_NAME**, **BATCH_SIZE** and **MAX_LEN**. 
WARNING: when using BERT model MAX_LEN can only be set to 512.
- **embeddings_models:**  
This section specifies the default model hyperparameters for the linear embeddings model, an option to run **GridSearchCV** with prefered parameter grid and the directory where the model will be saved.

### Running the application
To run the application, follow these steps:
1. Open a terminal or command prompt and navigate to the directory where the project is located.
2. Run the following command:
```python
python main.py
```
This will start the application using the default configuration.
### Changing configuration options
To override the default configuration options, you can pass command-line arguments to the **main.py** script. For example, to change the **modelHandler** option to **LinearFeaturesHandler**, you can use the following command:
```python
python main.py main.modelHandler=LinearFeaturesHandler
```

You can also **specify multiple configuration options** by separating them with a space, like this:
```python
python main.py main.modelHandler=LinearFeaturesHandler main.train=True
```
Note that the format for specifying configuration options is module.option=value, where module is the name of the module that contains the option, and option is the name of the option you want to change. The value is the new value you want to set for the option.

## Logging
Logging is set up using the setup_applevel_logger function and will log to a file named *app_debug.log*. The logger is set to log messages with a severity level of WARNING or higher, and the log message format includes the time, logger name, severity level, module name and line number, and the log message. The logger also includes a stream handler to output log messages to the console and a file handler to output log messages to a file, if specified.

## Contact
If you have any questions or issues, please don't hesitate to contact us.


