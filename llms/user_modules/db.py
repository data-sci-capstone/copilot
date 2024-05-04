from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from user_modules.table_instances import Base

# Setup database connection
engine = create_engine('postgresql+psycopg2://postgres:mypassword@copilot.craoqkiqslyh.us-east-2.rds.amazonaws.com:5432/copilot-db', echo=True)
Session = sessionmaker(bind=engine)