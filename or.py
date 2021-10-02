"""
author: Suresh
"""

from utils.model import	Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode = "a")

OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(OR)

# print(df)
logging.info(f"This is actual dataframe(df)")


X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10
try:
    logging.info(">>>> starting training  >>>>")
    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model,filename = "or.model")
    save_plot(df,"or.png" ,model)

    logging.info("<<<< training done successfully <<<<")
except Exception as e:
    logging.exception(e)
    raise e