from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DECIMAL, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class Models(Base):
    __tablename__ = 'models'
    model_id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False)
    gpu_usage = Column(Integer)
    eval_time = Column(Integer)

class Dialogues(Base):
    __tablename__ = "dialogues"
    dialogue_id = Column(Integer, primary_key=True)
    data_set = Column(String(3), nullable=False)
    dialogue_text = Column(Text, nullable=False)
    actual_summary = Column(Text, nullable=False)
    actual_sentiment = Column(String(8), nullable=True)

class Summaries(Base):
    __tablename__ = "summaries"
    summary_id = Column(Integer, primary_key=True)
    dialogue_id = Column(Integer, ForeignKey('dialogues.dialogue_id'))
    model_id = Column(Integer, ForeignKey('models.model_id'))
    generated_summary = Column(Text)
    rouge_score = Column(DECIMAL)
    meteor_score = Column(DECIMAL)
    bert_score = Column(DECIMAL)
    dialogue = relationship("Dialogues")
    model = relationship("Models")

class Sentiments(Base):
    __tablename__ = "sentiments"
    sentiment_id = Column(Integer, primary_key = True)
    dialogue_id = Column(Integer, ForeignKey('dialogues.dialogue_id'))
    model_id = Column(Integer, ForeignKey('models.model_id'))
    generated_sentiment = Column(String(8))

class SentimentEvaluation(Base):
    __tablename__ = "sentiment_evaluation"
    evaluation_id = Column(Integer, primary_key = True)
    model_id = Column(Integer, ForeignKey('models.model_id'))
    roc_score = Column(DECIMAL)
