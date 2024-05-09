from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DECIMAL, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class Models(Base):
    __tablename__ = 'models'
    model_id = Column(String(50), primary_key=True)
    gpu_usage = Column(Integer)
    eval_time = Column(Integer)

class Dialogues(Base):
    __tablename__ = "dialogues"
    dialogue_id = Column(Integer, primary_key=True)
    dataset = Column(String(10), nullable=False)
    dialogue_text = Column(Text, nullable=False)
    actual_summary = Column(Text, nullable=False)
    actual_sentiment = Column(String(8), nullable=True)

class Summaries(Base):
    __tablename__ = "summaries"
    summary_id = Column(Integer, primary_key=True)
    dialogue_id = Column(Integer, ForeignKey('dialogues.dialogue_id'))
    model_id = Column(String(50), ForeignKey('models.model_id'))
    generated_summary = Column(Text)
    rouge_1 = Column(DECIMAL)
    rouge_2 = Column(DECIMAL)
    rouge_l = Column(DECIMAL)
    bert_precision = Column(DECIMAL)
    bert_recall = Column(DECIMAL)
    bert_f1 = Column(DECIMAL)
    meteor = Column(DECIMAL)
    memory_summary_usage = Column(DECIMAL)
    time_summary_taken = Column(DECIMAL)
    dialogue = relationship("Dialogues")
    model = relationship("Models")

class Sentiments(Base):
    __tablename__ = "sentiments"
    sentiment_id = Column(Integer, primary_key = True)
    dialogue_id = Column(Integer, ForeignKey('dialogues.dialogue_id'))
    model_id = Column(String(50), ForeignKey('models.model_id'))
    generated_sentiment = Column(String(8))
    memory_sentiment_usage = Column(DECIMAL)
    time_sentiment_taken = Column(DECIMAL)

class SentimentEvaluation(Base):
    __tablename__ = "sentiment_evaluation"
    evaluation_id = Column(Integer, primary_key = True)
    model_id = Column(String(50), ForeignKey('models.model_id'))
    roc_score = Column(DECIMAL)
