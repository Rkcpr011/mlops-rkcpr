import pathlib
import os


current_directory = os.path.dirname(os.path.realpath(__file__)) #current directory of the script

PACKAGE_ROOT = os.path.dirname(current_directory) #parent directory of current directory


# PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

TARGET = 'Loan_Status'

#Final features used in the model
FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

# in our case it is same as Categorical features
FEATURES_TO_ENCODE = ['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Credit_History',
 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = 'CoapplicantIncome'

DROP_FEATURES = ['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount'] # taking log of numerical columns

# S3_BUCKET = "loanprediction"  #wiill not be used,instread local batch output variable will be used

LOCAL_BATCH_OUTPUT = "batch_outputs" 

# FOLDER="datadrift"

TRACKING_URI="https://dagshub.com/rakeshcpr011/MLOps-E2E-POC-i-mubahsir-hasan.mlflow"  #changed to localhost mlflow


EXPERIMENT_NAME="loan_prediction_model"

MODEL_NAME="Loanprediction-model"

