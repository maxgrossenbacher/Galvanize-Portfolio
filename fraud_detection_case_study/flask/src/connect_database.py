from sqlalchemy import create_engine
import pandas as pd
import psycopg2



def connect_psql(df, engine):
    df.to_sql('fraud_predictions', con=engine, if_exists='replace', index=False)

if __name__ == '__main__':
    streaming_data = pd.DataFrame({'a':[1,3],'b':[2,4], 'c':[1,2], 'd':[4,6]})
    engine = create_engine('postgresql+psycopg2://postgres@localhost:5432/fraud_predictions')
    connect_psql(streaming_data, engine)
