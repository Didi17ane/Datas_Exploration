import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

#################### connection à la base de données ###############
def get_data():
    try:
        conn = psycopg2.connect(
            database="tylim_db",
            user="didi_user",
            host="localhost",
            password="charb1709",
            port=5432,
        )
        query = "SELECT * FROM users"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except:
        st.error("Échec de connexion à la base de données.")

df = get_data()

if df.empty:
    st.warning("⚠️ Aucune donnée disponible.")
else:
   # Calcul des KPIs
    print(df)
    
    df.to_csv("../DATAS/DB_Tylimmo.csv", index=False, encoding="utf-8")
    