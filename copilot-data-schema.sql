CREATE DATABASE IF NOT EXISTS copilot-db;

USE copilot-db;

-- Dialogues Table
CREATE TABLE IF NOT EXISTS dialogues (
    dialogue_id SERIAL PRIMARY KEY, -- auto increments the id 
    dialogue_set VARCHAR(3) NOT NULL,
    dialogue_text TEXT NOT NULL,
    actual_summary TEXT NOT NULL,
    actual_sentiment VARCHAR(8)
);

-- Models Table
CREATE TABLE IF NOT EXISTS models (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL
);

-- Summaries Table
CREATE TABLE IF NOT EXISTS summaries (
    summary_id SERIAL PRIMARY KEY,
    dialogue_id INTEGER REFERENCES dialogues(dialogue_id),
    model_id INTEGER REFERENCES models(model_id),
    generated_summary TEXT,
    rouge_score DECIMAL,
    meteor_score DECIMAL
);

-- Sentiments Table
CREATE TABLE IF NOT EXISTS sentiments (
    sentiment_id SERIAL PRIMARY KEY,
    dialogue_id INTEGER REFERENCES dialogues(dialogue_id),
    model_id INTEGER REFERENCES models(model_id),
    generated_sentiment VARCHAR(8)
);

-- Evaluation Metrics for Sentiment Analysis
CREATE TABLE IF NOT EXISTS sentiment_evaluation (
    evaluation_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(model_id),
    model_name VARCHAR(50),
    roc_score DECIMAL
);