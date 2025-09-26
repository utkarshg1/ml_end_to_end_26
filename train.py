import joblib
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from constants import DATA_FILE, TARGET, TEST_SIZE, RANDOM_STATE, MODEL_DIR, MODEL_FILE


def train_and_save_model():
    try:
        # Write the info
        logger.info("Training Pipeline started")
        # Data ingestion
        logger.info("Data ingestion started")
        df = pd.read_csv(DATA_FILE)
        logger.info(f"Dataframe successfully loaded :\n{df.head()}")
        logger.info(f"Dataframe shape : {df.shape}")
        # Check for duplicates
        logger.info(f"Duplicate values : {df.duplicated().sum()}")
        # Drop the duplicates
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        logger.info(f"Shape after dropping duplicates : {df.shape}")
        # Missing values
        logger.info(f"Missing values :\n{df.isna().sum()}")
        logger.info(f"Data quality checks complete")
        # Seperating X and Y(target)
        logger.info(f"Seperating X and Y")
        X = df.drop(columns=[TARGET])
        Y = df[TARGET]
        logger.info("X and Y seperated")
        # Apply train test split
        xtrain, xtest, ytrain, ytest = train_test_split(
            X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logger.info(f"xtrain shape : {xtrain.shape}, ytrain shape : {ytrain.shape}")
        logger.info(f"xtest shape : {xtest.shape}, ytest shape : {ytest.shape}")
        # Apply preprocessing + logistic regression
        logger.info("Initializing preprocessing and model pipeline")
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(random_state=42),
        )
        # Cross validate model
        logger.info("Cross validating model")
        scores = cross_val_score(pipe, xtrain, ytrain, cv=5, scoring="f1_macro")
        cv_mean = scores.mean()
        cv_std = scores.std()
        logger.info(f"Cross validation scores f1_macro : {scores}")
        logger.info(f"Cross valdiation mean f1_macro : {cv_mean:.2%}")
        logger.info(f"Standard deviation scores : {cv_std:.2%}")
        # Fit the model
        logger.info(f"Fitting the model")
        pipe.fit(xtrain, ytrain)
        logger.info(f"Model fitting complete")
        # Evalutate model
        logger.info(f"Printing classification report")
        ypred_test = pipe.predict(xtest)
        logger.info(
            f"Classification Report :\n{classification_report(ytest, ypred_test)}"
        )
        logger.info("Model evaluation complete")
        # Save the model in joblib
        logger.info(f"Creating model directory : {MODEL_DIR}")
        MODEL_DIR.mkdir(exist_ok=True)
        logger.info(f"Saving the model object in : {MODEL_FILE}")
        joblib.dump(pipe, MODEL_FILE)
        logger.success(f"Training Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Exception occured : {e}")


if __name__ == "__main__":
    train_and_save_model()
