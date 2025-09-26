# use the model to predict any results
import joblib
import streamlit as st
from loguru import logger
from constants import MODEL_FILE
import pandas as pd


@st.cache_resource
def load_model(path=MODEL_FILE):
    try:
        logger.info(f"Loading model from : {path}")
        model = joblib.load(path)
        logger.info("Loading the model is successful")
        return model
    except Exception as e:
        logger.error(f"Exception occred while loading model : {e}")


def get_predicitons(model, sep_len, sep_wid, pet_len, pet_wid):
    try:
        logger.info("Loading as dataframe")
        xnew = pd.DataFrame(
            [
                {
                    "sepal_length": sep_len,
                    "sepal_width": sep_wid,
                    "petal_length": pet_len,
                    "petal_width": pet_wid,
                }
            ]
        )
        logger.info(f"Dataframe loaded successfully :\n{xnew}")
        logger.info("Predciting results ")
        preds = model.predict(xnew)[0]
        logger.info(f"Predicted species : {preds}")
        probs = model.predict_proba(xnew).round(4)
        probs_df = pd.DataFrame(probs, columns=model.classes_)
        logger.info(f"Predicted probabilities :\n{probs_df}")
        return preds, probs_df
    except Exception as e:
        logger.error(f"Exception occured during inference : {e}")


if __name__ == "__main__":
    model = load_model()
    preds, probs = get_predicitons(model, sep_len=4, sep_wid=2, pet_len=3, pet_wid=1)
