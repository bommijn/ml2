from pathlib import Path
import os

class Config:
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "evaluation_results")
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    
    # Data files
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, "Titanic-Dataset.csv")
    
    # Logging configuration
    LOG_FILE = os.path.join(LOG_DIR, "titanic_pipeline.log")
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_LEVEL = "INFO"
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Features configuration
    FEATURES = [
        'Pclass',
        'Age',
        'Fare',
        'Sex_female',
        'Sex_male',
        'Embarked_C',
        'Embarked_Q',
        'Embarked_S'
    ]
    
    COLUMNS_TO_DROP = [
        'PassengerId',
        'Name',
        'Ticket',
        'Cabin'
    ]
    
    # Model hyperparameters for grid search
    XGB_PARAMS = {
        'max_depth': range(2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05]
    }
    
    # Cross validation settings
    CV_FOLDS = 5
    N_JOBS = -1 
    
    @staticmethod
    def create_directories():
        """Create all necessary directories if they don't exist."""
        for directory in [Config.DATA_DIR, Config.OUTPUT_DIR, Config.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)