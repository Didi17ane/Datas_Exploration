import streamlit as st
import matplotlib.pyplot as plt


def revenu(data):

    # Revenu moyen par tranche d’âge
    fig, ax = plt.subplots()
    data.groupby("age_grp")["mean_rev"].mean().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("Revenu moyen (FCFA)")

    return fig

def carte(data):

    df_region = data.groupby("region")["mean_rev"].mean().reset_index()
            
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=7.54, longitude=-5.55, zoom=6),
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                data="../geoBoundaries-CIV-ADM1.geojson",
                get_fill_color="[255, (1-mean_rev/200000)*255, 0]",  # gradient selon revenu
                pickable=True,
            ),
        ],
    ))

def logement(data):

    # Statuts logement moyens nationaux
    st.subheader("Répartition statuts logement (nation)")
    statut_cols = ["proprio_titre","proprio_sans","locataire","autre_logement"]
    st.bar_chart(data[statut_cols].mean())

    st.metric("Assurance maladie", f"{round(100*data['assurance'].mean(),1)} %")
