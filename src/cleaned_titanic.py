import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import logging
import joblib
from config import Config

#do this first otherwise it might crash
Config.create_directories()


# Set up logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    filename=Config.LOG_FILE
)
logger = logging.getLogger(__name__)

def load_data(filepath: str = None):
    """
    Load and preprocess the titanic dataset.
    """
    if filepath is None:
        filepath = Config.TRAIN_DATA_PATH
        
    try:
        logger.info("Loading data from %s", filepath)
        df = pd.read_csv(filepath)
        return df
        
    except Exception as e:
        logger.error("Error in loading data: %s", str(e))
        raise

def preprocess_data(data: pd.DataFrame, test_size: float = None, random_state: int = None):
    '''
    Process the data by dropping unnecessary columns, handling categorical variables.
    '''
    # use the config values if parameters are not provided
    test_size = test_size or Config.TEST_SIZE
    random_state = random_state or Config.RANDOM_STATE
            
    # drop unnecessary columns
    data.drop(Config.COLUMNS_TO_DROP, axis=1, inplace=True)
            
    # handle categorical variables
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'])
            
    # handle missing values
    data.dropna(inplace=True)
            
    # define features from config
    features = Config.FEATURES

    #split features and target
    X = data[features]
    y = data['Survived']
             
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # scale the data
    scaler = StandardScaler()
    X_train[Config.NUMERIC_FEATURES] = scaler.fit_transform(X_train[Config.NUMERIC_FEATURES])
    X_test[Config.NUMERIC_FEATURES] = scaler.transform(X_test[Config.NUMERIC_FEATURES]) 
    
    logger.info("Data preprocessing completed successfully")
    return X_train, X_test, y_train, y_test, scaler

def train_models(X_train: np.ndarray, y_train: np.ndarray, random_state: int = None):
    """
    Train multiple models and create an ensemble.
    """
    try:
        random_state = random_state or Config.RANDOM_STATE
        logger.info("Starting model training")
        
        # train Random Forest
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(X_train, y_train)
        logger.info("Random Forest training completed")
        
        # train XGBoost with gridSearch using config parameters
        xgb_base = XGBClassifier(random_state=random_state)
        grid_search = GridSearchCV(
            xgb_base,
            Config.XGB_PARAMS,
            n_jobs=Config.N_JOBS,
            cv=Config.CV_FOLDS
        )
        grid_search.fit(X_train, y_train)
        
        # get best xgboost model
        xgb_model = XGBClassifier(random_state=random_state, **grid_search.best_params_)
        xgb_model.fit(X_train, y_train)
        logger.info("XGBoost training completed with best params: %s", grid_search.best_params_)
        
        # create voting classifier
        voting_clf = VotingClassifier(
            estimators=[('rf', rf_model), ('xgb', xgb_model)],
            voting='soft'
        )
        voting_clf.fit(X_train, y_train)
        logger.info("Ensemble model training completed")
        
        return {
            'random_forest': rf_model,
            'xgboost': xgb_model,
            'ensemble': voting_clf
        }
        
    except Exception as e:
        logger.error("Error in model training: %s", str(e))
        raise

def evaluate_models(models: dict, X_test: np.ndarray, y_test: np.ndarray, output_dir: str = None):
    """
    Evaluate trained models and save visualization results.
    """
    try:
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR
            
        logger.info("Starting model evaluation")
        evaluation_results = {}
        
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            
            #calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            evaluation_results[model_name] = metrics
            
            logger.info(f"\nMetrics for {model_name}:")
            logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            logger.info(f"Precision: {metrics['precision']:.3f}")
            
            # save confusion matrix plot
            if output_dir:
                plt.figure(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=metrics['confusion_matrix'])
                disp.plot()
                plt.title(f'Confusion Matrix - {model_name}')
                plt.savefig(f'{output_dir}/{model_name}_confusion_matrix.png')
                plt.close()
        
        return evaluation_results
        
    except Exception as e:
        logger.error("Error in model evaluation: %s", str(e))
        raise

def main(data_filepath: str = None, output_dir: str = None):
    """
    Main function to run the complete pipeline.
    """
    try:
        # Use config values if not provided
        data_filepath = data_filepath or Config.TRAIN_DATA_PATH
        output_dir = output_dir or Config.OUTPUT_DIR
        
        # kload and preprocess data
        df = load_data(data_filepath)
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
        
        # train models
        trained_models = train_models(X_train_scaled, y_train)
        
        # evaluate models
        evaluation_results = evaluate_models(trained_models, X_test_scaled, y_test, output_dir)
        
        return trained_models, evaluation_results, scaler
        
    except Exception as e:
        logger.error("Error in pipeline execution: %s", str(e))
        raise

def save_model(model, model_name):
    """
    Save the trained models and scaler to disk.
    """
    try:
        model_path = f"{Config.MODEL_DIR}\{model_name}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # save scaler
        scaler_path = f"{Config.MODEL_DIR}/scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
    except Exception as e:
        logger.error("Error in saving model: %s", str(e))
        raise


if __name__ == "__main__":   
    trained_models, evaluation_results, scaler = main()
    save_model(trained_models['random_forest'], 'random_forest')
    logger.info("Pipeline execution completed successfully")
    