import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Configuration de la page
st.set_page_config(
    page_title="Bénin Geo-Watch",
    page_icon="🇧🇯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour donner un aspect premium
st.markdown("""
<style>
    .kpi-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #2ecc71;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 32px;
        font-weight: bold;
        color: white;
    }
    .kpi-label {
        font-size: 14px;
        color: #B0B0B0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .insight-card {
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-top: 3px solid #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        # Load datasets
        events_df = pd.read_csv("../data/events_clean.csv")
        daily_score_df = pd.read_csv("../data/daily_score.csv")
        
        # Format dates
        events_df['SQLDATE'] = pd.to_datetime(events_df['SQLDATE'])
        events_df['month_name'] = events_df['SQLDATE'].dt.strftime('%Y-%m')
        
        daily_score_df['SQLDATE'] = pd.to_datetime(daily_score_df['SQLDATE'])
        daily_score_df['month_name'] = daily_score_df['SQLDATE'].dt.strftime('%Y-%m')
        
        return events_df, daily_score_df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame(), pd.DataFrame()

events_df, daily_score_df = load_data()

if events_df.empty:
    st.stop()

# --- HEADER ---
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<h1 style='text-align: center;'>🇧🇯</h1>", unsafe_allow_html=True)
with col_title:
    st.title("Bénin Geo-Watch : Radar Géopolitique et Sécuritaire")
    st.markdown("*À partir des données GDELT 2025/2026*")

st.divider()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filtres")
months = ["Tous"] + sorted(list(events_df['month_name'].dropna().unique()))
selected_month = st.sidebar.selectbox("Filtrer par mois", months)

event_types = sorted(list(events_df['EventType'].dropna().unique()))
selected_event_types = st.sidebar.multiselect(
    "Type d'événement",
    options=event_types,
    default=event_types
)

# Apply filters
filtered_events = events_df.copy()
if selected_month != "Tous":
    filtered_events = filtered_events[filtered_events['month_name'] == selected_month]
if selected_event_types:
    filtered_events = filtered_events[filtered_events['EventType'].isin(selected_event_types)]

# --- KPIs ---
total_events = len(filtered_events)
avg_goldstein = filtered_events['GoldsteinScale'].mean()
avg_tone = filtered_events['AvgTone'].mean()
# Using daily score for stability
avg_stability = daily_score_df['stability_score'].mean() if 'stability_score' in daily_score_df.columns else 53

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total d'événements</div>
        <div class="kpi-value">{total_events:,}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    color = "#2ecc71" if avg_goldstein > 0 else "#e74c3c"
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: {color}">
        <div class="kpi-label">Goldstein Moyen</div>
        <div class="kpi-value" style="color: {color}">{avg_goldstein:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    color = "#2ecc71" if avg_tone > 0 else "#e74c3c"
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: {color}">
        <div class="kpi-label">Tone Médiatique Moyen</div>
        <div class="kpi-value" style="color: {color}">{avg_tone:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: #3498db">
        <div class="kpi-label">Score de Stabilité</div>
        <div class="kpi-value" style="color: #3498db">{avg_stability:.0f}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# --- CARTE GÉOGRAPHIQUE ---
st.subheader("📍 Carte des Tensions Régionales")
st.markdown("*Vue géographique des événements géolocalisés. La taille de la bulle indique le **volume d'événements**, et la couleur indique le **sentiment** (rouge = négatif/crise, vert = positif/coopération).*")

def extract_region(geo_name):
    if not isinstance(geo_name, str): return "National"
    parts = geo_name.split(", ")
    if len(parts) > 1:
        region = parts[1]
        if region == "Qué": return "Ouémé"
        if region == "Kouffo": return "Couffo"
        if region == "Atakora": return "Atacora"
        return region
    return "National"

filtered_events['Region'] = filtered_events['ActionGeo_FullName'].apply(extract_region)

benin_coords = {
    'Alibori': {'lat': 11.0, 'lon': 2.5},
    'Atacora': {'lat': 10.5, 'lon': 1.5},
    'Atlantique': {'lat': 6.5, 'lon': 2.2},
    'Borgou': {'lat': 9.5, 'lon': 2.5},
    'Collines': {'lat': 8.0, 'lon': 2.0},
    'Donga': {'lat': 9.2, 'lon': 1.6},
    'Couffo': {'lat': 7.0, 'lon': 1.8},
    'Littoral': {'lat': 6.35, 'lon': 2.4},
    'Mono': {'lat': 6.6, 'lon': 1.8},
    'Ouémé': {'lat': 6.5, 'lon': 2.6},
    'Plateau': {'lat': 7.0, 'lon': 2.6},
    'Zou': {'lat': 7.2, 'lon': 2.0}
}

regional_data = filtered_events[filtered_events['Region'].isin(benin_coords.keys())]

if not regional_data.empty:
    reg_stats = regional_data.groupby('Region').agg(
        Nb_Events=('SQLDATE', 'count'),
        Avg_Tone=('AvgTone', 'mean')
    ).reset_index()
    
    reg_stats['lat'] = reg_stats['Region'].map(lambda x: benin_coords[x]['lat'])
    reg_stats['lon'] = reg_stats['Region'].map(lambda x: benin_coords[x]['lon'])
    
    fig_map = px.scatter_mapbox(
        reg_stats, lat="lat", lon="lon", size="Nb_Events", color="Avg_Tone",
        hover_name="Region", hover_data={"lat": False, "lon": False, "Nb_Events": True, "Avg_Tone": ':.2f'},
        color_continuous_scale="RdYlGn", size_max=40, zoom=6,
        center={"lat": 8.5, "lon": 2.2},
        mapbox_style="carto-darkmatter"
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Aucun événement géolocalisé au niveau régional pour cette sélection.")

st.divider()

# --- CHARTS SECTION ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Événements et Tone Médiatique")
    st.markdown("*Ce graphique croise le volume d'articles publiés (bruit médiatique) avec le **sentiment général** (Tone). Une chute abrupte de la courbe rouge est notre premier signal d'alerte d'une crise.*")
    monthly_stats = filtered_events.groupby('month_name').agg(
        Nb_Events=('SQLDATE', 'count'),
        Avg_Tone=('AvgTone', 'mean')
    ).reset_index()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=monthly_stats['month_name'], y=monthly_stats['Nb_Events'], name="Nb Événements", marker_color='#3498db'))
    fig1.add_trace(go.Scatter(x=monthly_stats['month_name'], y=monthly_stats['Avg_Tone'], name="Tone Moyen", yaxis="y2", line=dict(color='#e74c3c', width=3)))
    fig1.update_layout(template="plotly_dark", yaxis=dict(title="Nb Événements"), yaxis2=dict(title="Tone Moyen", overlaying="y", side="right"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

with row1_col2:
    st.subheader("Intensité des Conflits (Goldstein Scale)")
    st.markdown("*L'échelle de Goldstein mesure le potentiel de déstabilisation. Plus la courbe plonge, plus la probabilité de violences physiques ou politiques est élevée. C'est l'indicateur d'alerte principal.*")
    monthly_goldstein = filtered_events.groupby('month_name')['GoldsteinScale'].mean().reset_index()
    fig2 = px.area(monthly_goldstein, x='month_name', y='GoldsteinScale', template="plotly_dark", color_discrete_sequence=['#9b59b6'])
    fig2.update_layout(xaxis_title="", yaxis_title="Goldstein Moyen")
    st.plotly_chart(fig2, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Distribution des Événements par Type")
    st.markdown("*Permet de nuancer l'analyse : une période de crise n'exclut pas une forte activité diplomatique (Coopération) en parallèle.*")
    top_events = filtered_events['EventType'].value_counts().head(10).reset_index()
    top_events.columns = ['EventType', 'Count']
    fig3 = px.pie(top_events, values='Count', names='EventType', template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig3, use_container_width=True)

with row2_col2:
    st.subheader("Évolution de la Stabilité Globale")
    st.markdown("*Ce score agrégé quotidiennement montre la 'santé géopolitique' du pays. La courbe rouge permet de lisser les données pour dégager la tendance de fond sur 7 jours.*")
    if not daily_score_df.empty and 'stability_score' in daily_score_df.columns:
        fig4 = px.line(daily_score_df, x='SQLDATE', y='stability_score', template="plotly_dark", color_discrete_sequence=['#2ecc71'], title="")
        if len(daily_score_df) > 7:
            daily_score_df['rolling_stability'] = daily_score_df['stability_score'].rolling(window=7).mean()
            fig4.add_trace(go.Scatter(x=daily_score_df['SQLDATE'], y=daily_score_df['rolling_stability'], mode='lines', name='Moyenne 7 jours', line=dict(color='#e74c3c', width=2)))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Données de stabilité non disponibles.")

st.divider()

# --- INSIGHTS ---
st.header("💡 Insights Clés & Analyse")
i_col1, i_col2, i_col3 = st.columns(3)
with i_col1:
    st.markdown("""<div class="insight-card"><h4>🚨 Pic d'Instabilité</h4><p>En décembre, le nombre d'événements (près de 1100) a représenté le double de la moyenne mensuelle. Un signal d'instabilité majeur qui a déclenché nos alertes de crise.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="insight-card"><h4>📰 Dramatisation Médiatique</h4><p>Les articles couvrant une crise affichent un ton moyen de <b>-1.72</b> contre +1.13 pour les autres. La presse internationale polarise fortement l'actualité sécuritaire.</p></div>""", unsafe_allow_html=True)
with i_col2:
    st.markdown("""<div class="insight-card" style="border-top-color: #2ecc71;"><h4>🤝 Résilience Diplomatique</h4><p>Malgré un contexte sécuritaire tendu au Nord, plus de <b>51%</b> des événements recensés sont de type "Coopération". L'activité diplomatique du Bénin reste extrêmement solide.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="insight-card" style="border-top-color: #3498db;"><h4>🇳🇬 Le Poids du Nigeria</h4><p>Avec plus de 600 événements conjoints, le Nigeria s'affirme de très loin comme le premier acteur d'interaction géopolitique du Bénin devant la France et les autres pays frontaliers.</p></div>""", unsafe_allow_html=True)
with i_col3:
    st.markdown("""<div class="insight-card" style="border-top-color: #f1c40f;"><h4>📉 Le Mois de Tous les Dangers</h4><p>L'échelle de Goldstein montre que le mois de novembre a été le mois où l'intensité moyenne des conflits a été la plus critique (score descendant à +0.13, frôlant le négatif).</p></div>""", unsafe_allow_html=True)
