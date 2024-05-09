CREATE DATABASE IF NOT EXISTS copilot-db;

USE copilot-db;

-- Dialogues Table
CREATE TABLE IF NOT EXISTS dialogues (
    dialogue_id SERIAL PRIMARY KEY, -- "serial" auto increments the id 
    dataset VARCHAR(10) NOT NULL,
    dialogue_text TEXT NOT NULL,
    actual_summary TEXT NOT NULL,
    actual_sentiment VARCHAR(8)
);

-- Models Table
CREATE TABLE IF NOT EXISTS models (
    model_id VARCHAR(50) PRIMARY KEY,
    gpu_usage INTEGER,
    eval_time INTEGER
);

-- Summaries Table
CREATE TABLE IF NOT EXISTS summaries (
    summary_id SERIAL PRIMARY KEY,
    dialogue_id INTEGER REFERENCES dialogues(dialogue_id),
    model_id VARCHAR(50) REFERENCES models(model_id),
    generated_summary TEXT,
    rouge_1 DECIMAL,
    rouge_2 DECIMAL,
    rouge_l DECIMAL,
    bert_precision DECIMAL,
    bert_recall DECIMAL,
    bert_f1 DECIMAL,
    meteor DECIMAL,
    memory_summary_usage DECIMAL,
    time_summary_taken DECIMAL
);

-- Sentiments Table
CREATE TABLE IF NOT EXISTS sentiments (
    sentiment_id SERIAL PRIMARY KEY,
    dialogue_id INTEGER REFERENCES dialogues(dialogue_id),
    model_id VARCHAR(50) REFERENCES models(model_id),
    generated_sentiment VARCHAR(8),
    memory_sentiment_usage DECIMAL,
    time_sentiment_taken DECIMAL
);

-- Evaluation Metrics for Sentiment Analysis
CREATE TABLE IF NOT EXISTS sentiment_evaluation (
    evaluation_id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models(model_id),
    accuracy DECIMAL,
    f1_score DECIMAL
);