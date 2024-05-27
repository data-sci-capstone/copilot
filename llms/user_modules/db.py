from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from table_instances import Base, Sentiments, Summaries, Models, SentimentEvaluation, Dialogues
import pandas as pd

# Setup database connection
engine = create_engine('postgresql+psycopg2://postgres:mypassword@copilot.craoqkiqslyh.us-east-2.rds.amazonaws.com:5432/copilot-db')
Session = sessionmaker(bind=engine)

def get_data(dataset: str)->pd.DataFrame:
    data = pd.read_sql(f"SELECT * FROM dialogues WHERE dataset = '{dataset}';", engine)
    return data

def get_training_data()->pd.DataFrame:
    data = pd.read_sql(f"SELECT * FROM dialogues WHERE dataset = 'training';", engine)
    return data

def get_validation_data()->pd.DataFrame:
    validation = pd.read_csv("SELECT * FROM dialogues WHERE dataset = 'validation';", engine)
    return validation

def get_test_data()->pd.DataFrame:
    test = pd.read_csv("SELECT * FROM dialogues WHERE dataset = 'test';", engine)
    return test

def get_model_sentiment_data(model_id: str)->pd.DataFrame:
    model_data = pd.read_sql(f"SELECT * FROM sentiments WHERE model_id = '{model_id}';", engine)
    return model_data

def get_mistral_sentiment_data()->pd.DataFrame:
    mistral_sentiment = pd.read_sql(f"SELECT * FROM sentiments WHERE model_id = 'mistral 7b';", engine)
    return mistral_sentiment

def get_gemma_sentiment_data()->pd.DataFrame:
    gemma_sentiment = pd.read_sql(f"SELECT * FROM sentiments WHERE model_id = 'gemma 7b';", engine)
    return gemma_sentiment

def get_llama3_sentiment_data()->pd.DataFrame:
    llama3_sentiment = pd.read_sql(f"SELECT * FROM sentiments WHERE model_id = 'llama 3 8b';", engine)
    return llama3_sentiment

def add_models_data(df: pd.DataFrame) -> None:

    if not column_error(list(df.columns), 'models'):
        session = Session()
        dict_converted_models = df.to_dict(orient='records')
        session.bulk_insert_mappings(Models, dict_converted_models)
        session.commit()
        session.close()

def add_dialogues_data(df: pd.DataFrame) -> None:

    if not column_error(list(df.columns), 'dialogues'):
        session = Session()
        dict_converted_dialogues = df.to_dict(orient='records')
        session.bulk_insert_mappings(Dialogues, dict_converted_dialogues)
        session.commit()
        session.close()

def add_summaries_data(df: pd.DataFrame) -> None:

    if not column_error(list(df.columns), 'summaries'):
        session = Session()
        dict_converted_summaries = df.to_dict(orient='records')
        session.bulk_insert_mappings(Summaries, dict_converted_summaries)
        session.commit()
        session.close()

def add_sentiments_data(df: pd.DataFrame) ->None:

    if not column_error(list(df.columns), 'sentiments'):
        session = Session()
        dict_converted_sentiments = df.to_dict(orient='records')
        session.bulk_insert_mappings(Sentiments, dict_converted_sentiments)
        session.commit()
        session.close()

def add_sentiment_evaluation_data(df: pd.DataFrame) -> None:

    if not column_error(list(df.columns), 'sentiment_evaluation'):
        session = Session()
        dict_converted_sentiment_evaluation = df.to_dict(orient='records')
        session.bulk_insert_mappings(SentimentEvaluation, 
            dict_converted_sentiment_evaluation)
        session.commit()
        session.close()

def column_error(df_cols: list, table_name: str) -> bool:

    correct_column_names = {
        'models': {
            'model_id', 'gpu_usage', 'eval_time'
            },
        'dialogues': {
            'data_set', 'dialogue_text', 'actual_summary', 'actual_sentiment'
            },
        'summaries': {
            'dialogue_id', 'model_id', 'generated_summary', 'rouge_1', 
            'rouge_2','rouge_l', 'bert_precision', 'bert_recall', 
            'bert_f1', 'meteor','memory_summary_usage', 
            'time_summary_taken'
            },
        'sentiments': {
            'dialogue_id', 'model_id', 'generated_sentiment', 
            'memory_sentiment_usage', 'time_sentiment_taken'
            },
        'sentiment_evaluation': {
            'model_id', 'accuracy', 'f1_score'
            }
    }

    for col in df_cols:
        if col not in correct_column_names[table_name]:
            print(f"ERROR: df col: {col} does not match db {table_name} \
            column equivalent")
            return True
    return False