import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import pickle
import json
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Chargement de la clé API :
#   1. En local  → fichier .env à la racine du projet (jamais pushé)
#   2. En prod   → st.secrets (configuré dans les Settings de Streamlit Cloud)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
else:
    # Fallback Streamlit Cloud : charge depuis st.secrets si disponible
    try:
        for _k, _v in st.secrets.items():
            os.environ.setdefault(_k, str(_v))
    except Exception:
        pass  # st.secrets non disponible (ex: test local sans .env ni secrets)

# Configuration de la page
st.set_page_config(
    page_title="Bénin Geo-Watch",
    page_icon="BJ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour donner un aspect premium + verrouillage du thème sombre
st.markdown("""
<style>
    /* Cache le toolbar Streamlit (sélecteur de thème, deploy, etc.) */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
    }
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
        base_path = Path(__file__).parent.parent
        events_path = base_path / "data" / "events_clean.csv"
        daily_score_path = base_path / "data" / "daily_score.csv"
        
        # Format dates
        events_df = pd.read_csv(events_path, low_memory=False)
        daily_score_df = pd.read_csv(daily_score_path, low_memory=False)
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
    st.markdown("<div style='text-align:center; font-size:48px; font-weight:bold; color:#2ecc71;'>BJ</div>", unsafe_allow_html=True)
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

# ============================================================
# --- RAG ASSISTANT ---
# ============================================================

@st.cache_resource(show_spinner="Chargement de la base vectorielle…")
def load_rag_components():
    """Charge l'index FAISS, les métadonnées et le client Mistral (une seule fois)."""
    try:
        import faiss
        from mistralai.client import Mistral

        # Résolution robuste du chemin : on cherche depuis __file__ (absolu) puis depuis cwd
        _app_file = Path(__file__).resolve()
        _candidates = [
            _app_file.parent.parent / "data" / "VectorBase_db",   # lancé depuis dashboard/
            Path.cwd() / "data" / "VectorBase_db",                 # lancé depuis racine projet
            Path.cwd().parent / "data" / "VectorBase_db",          # autre contexte
        ]
        base = next((p for p in _candidates if (p / "benin_events.index").exists()), None)
        if base is None:
            return None, None, None

        index = faiss.read_index(str(base / "benin_events.index"))
        with open(base / "events_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        api_key = os.environ.get("MISTRAL_API_KEY", "")
        client = Mistral(api_key=api_key)
        return index, metadata, client
    except Exception:
        return None, None, None


def chat_with_rag(query: str, index, metadata, mistral_client) -> tuple[str, list[str]]:
    """Embed la question, récupère le contexte FAISS, génère la réponse Mistral."""
    # 1. Embedding de la question
    res_query = mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=[query]
    )
    query_vec = np.array([res_query.data[0].embedding]).astype("float32")

    # 2. Recherche Top-2 dans FAISS
    _D, I = index.search(query_vec, k=2)

    # 3. Construction du contexte
    context = ""
    sources = []
    for idx in I[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        article = metadata[idx]
        context += f"\n---\nARTICLE ({article.get('EventType', '?')}): {article.get('article_text', '')}\n"
        url = article.get("SOURCEURL", "")
        if url:
            sources.append(url)

    # 4. Prompt
    prompt = f"""Tu es un expert en actualité béninoise. Réponds à la question en utilisant UNIQUEMENT le contexte fourni ci-dessous.
Si la réponse n'est pas dans le contexte, dis-le poliment.

CONTEXTE:
{context}

QUESTION:
{query}

RÉPONSE:"""

    # 5. Appel Mistral Chat
    chat_response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat_response.choices[0].message.content, list(set(sources))


# --- Initialisation de l'historique de conversation ---
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []  # liste de {"role": "user"|"bot", "content": str}

# --- Chargement des composants RAG ---
faiss_index, events_metadata, mistral_client_rag = load_rag_components()

# --- UI Sidebar RAG ---
st.sidebar.divider()
st.sidebar.markdown(
    "<h3 style='margin-bottom:4px;'>Assistant Bénin Geo-Watch</h3>"
    "<p style='font-size:12px; color:#888; margin-top:0;'>Posez une question sur les données géopolitiques du Bénin</p>",
    unsafe_allow_html=True
)

if faiss_index is None:
    st.sidebar.error("❌ Base vectorielle introuvable. Vérifiez `data/VectorBase_db/`.")
else:
    # Affichage de l'historique
    for msg in st.session_state.rag_history:
        if msg["role"] == "user":
            st.sidebar.markdown(
                f"<div style='background:#2a2a2a;border-left:3px solid #3498db;padding:8px 10px;"
                f"border-radius:6px;margin-bottom:6px;font-size:13px;'>"
                f"<b style='color:#3498db;'>Vous</b><br>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                f"<div style='background:#1e1e1e;border-left:3px solid #2ecc71;padding:8px 10px;"
                f"border-radius:6px;margin-bottom:6px;font-size:13px;'>"
                f"<b style='color:#2ecc71;'>Assistant</b><br>{msg['content']}</div>",
                unsafe_allow_html=True
            )
            if msg.get("sources"):
                with st.sidebar.expander("📎 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"- [{src[:60]}…]({src})" if len(src) > 60 else f"- [{src}]({src})")

    # Saisie de la question
    user_question = st.sidebar.text_area(
        "Votre question :",
        placeholder="Ex: Que s'est-il passé dans le Borgou en décembre ?",
        height=90,
        key="rag_input"
    )
    send_col, clear_col = st.sidebar.columns([2, 1])
    send_btn = send_col.button("📨 Envoyer", use_container_width=True, type="primary")
    clear_btn = clear_col.button("🗑️ Effacer", use_container_width=True)

    if clear_btn:
        st.session_state.rag_history = []
        st.rerun()

    if send_btn and user_question.strip():
        st.session_state.rag_history.append({"role": "user", "content": user_question.strip()})
        with st.sidebar:
            with st.spinner("Recherche en cours…"):
                try:
                    answer, sources = chat_with_rag(
                        user_question.strip(),
                        faiss_index,
                        events_metadata,
                        mistral_client_rag
                    )
                    st.session_state.rag_history.append({
                        "role": "bot",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.session_state.rag_history.append({
                        "role": "bot",
                        "content": f"⚠️ Erreur : {e}",
                        "sources": []
                    })
        st.rerun()

# --- GLOSSAIRE DES INDICATEURS ---
with st.expander("Comprendre les indicateurs GDELT", expanded=False):
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("""
**Echelle de Goldstein** (-10 → +10)
> Mesure le *potentiel de déstabilisation* d'un événement. Un score **négatif** signale une menace pour la stabilité (conflit, violence, sanction). Un score **positif** indique un événement coopératif (accord, aide, diplomatie). C'est l'**indicateur d'alerte principal** de ce dashboard.

**Tone Médiatique (AvgTone)**
> Sentiment émotionnel moyen de tous les articles couvrant un événement. Calculé par GDELT à partir de l'analyse linguistique des textes. Valeur **négative** = presse alarmiste ou critique. Valeur **positive** = presse favorable ou enthousiaste.

**Score de Stabilité** (0 → 100)
> Score composite quotidien dérivé du Goldstein et du Tone. **50 = neutre**, en dessous = instabilité, au-dessus = bonne santé géopolitique. Utilisé pour les prédictions temporelles.
""")
    with g_col2:
        st.markdown("""
**Nombre de Mentions (NumMentions)**
> Combien de fois un événement est cité à travers l'ensemble des flux d'information GDELT. Une valeur élevée signifie que l'événement a généré un important **bruit médiatique**.

**Nombre de Sources (NumSources)**
> Nombre de sites d'information *distincts* qui ont couvert l'événement. Indique la **diversité** de la couverture médiatique. Un événement couverts par 1 seule source est moins fiable/significatif qu'un événement relayé par 20 sources indépendantes.

**Nombre d'Articles (NumArticles)**
> Nombre total d'articles publiés. Proche de NumMentions mais compte les articles complets plutôt que les citations. Combiné avec NumSources, il mesure l'**ampleur médiatique** réelle d'un événement.
""")

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
st.subheader("Carte des Tensions Régionales")
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
st.header("Insights Clés & Analyse")
i_col1, i_col2, i_col3 = st.columns(3)
with i_col1:
    st.markdown("""<div class="insight-card"><h4>Pic d'Instabilité</h4><p>En décembre, le nombre d'événements (près de 1100) a représenté le double de la moyenne mensuelle. Un signal d'instabilité majeur qui a déclenché nos alertes de crise.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="insight-card"><h4>Dramatisation Médiatique</h4><p>Les articles couvrant une crise affichent un ton moyen de <b>-1.72</b> contre +1.13 pour les autres. La presse internationale polarise fortement l'actualité sécuritaire.</p></div>""", unsafe_allow_html=True)
with i_col2:
    st.markdown("""<div class="insight-card" style="border-top-color: #2ecc71;"><h4>Résilience Diplomatique</h4><p>Malgré un contexte sécuritaire tendu au Nord, plus de <b>51%</b> des événements recensés sont de type "Coopération". L'activité diplomatique du Bénin reste extrêmement solide.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="insight-card" style="border-top-color: #3498db;"><h4>Le Poids du Nigeria</h4><p>Avec plus de 600 événements conjoints, le Nigeria s'affirme de très loin comme le premier acteur d'interaction géopolitique du Bénin devant la France et les autres pays frontaliers.</p></div>""", unsafe_allow_html=True)
with i_col3:
    st.markdown("""<div class="insight-card" style="border-top-color: #f1c40f;"><h4>Le Mois de Tous les Dangers</h4><p>L'échelle de Goldstein montre que le mois de novembre a été le mois où l'intensité moyenne des conflits a été la plus critique (score descendant à +0.13, frôlant le négatif).</p></div>""", unsafe_allow_html=True)

st.divider()

# ============================================================
# --- SIMULATEUR WHAT-IF ---
# ============================================================

st.header("Simulateur What-If : Analysez un Scénario")
st.markdown("*Ajustez les paramètres géopolitiques ci-dessous pour simuler n'importe quel scénario et découvrir comment notre modèle le classerait.*")

@st.cache_resource(show_spinner="Chargement des modèles ML…")
def load_ml_models():
    """Charge les modèles KMeans + Scaler + règles de cluster."""
    try:
        _base = Path(__file__).resolve().parent.parent / "models"
        _candidates = [
            _base,
            Path.cwd() / "models",
            Path.cwd().parent / "models",
        ]
        model_dir = next((p for p in _candidates if (p / "kmeans_geopolitics.pkl").exists()), None)
        if model_dir is None:
            return None, None, None

        kmeans  = joblib.load(model_dir / "kmeans_geopolitics.pkl")
        scaler  = joblib.load(model_dir / "scaler_geopolitics.pkl")
        regles  = json.loads((model_dir / "regles_du_modele.json").read_text(encoding="utf-8"))
        return kmeans, scaler, regles
    except Exception:
        return None, None, None

_CLUSTER_COLORS = {
    "Crise/Conflit Négatif":           "#e74c3c",
    "Événement Fortement Médiatisé":   "#f39c12",
    "Tension Modérée":                 "#f1c40f",
    "Coopération/Diplomatie Positive": "#2ecc71",
    "Événement Neutre/Routinier":      "#3498db",
}
_CLUSTER_ICONS = {
    "Crise/Conflit Négatif":           "ALERTE",
    "Événement Fortement Médiatisé":   "MÉDIATISÉ",
    "Tension Modérée":                 "TENSION",
    "Coopération/Diplomatie Positive": "COOPÉRATION",
    "Événement Neutre/Routinier":      "NEUTRE",
}

kmeans_model, scaler_model, cluster_regles = load_ml_models()

if kmeans_model is None:
    st.error("❌ Modèles ML introuvables. Vérifiez le dossier `models/`.")
else:
    sim_col1, sim_col2 = st.columns([1, 1], gap="large")

    with sim_col1:
        st.subheader("Parametres du scénario")
        st.caption("Faites glisser les curseurs pour décrire votre scénario. La classification se met à jour en temps réel.")

        goldstein = st.slider(
            "Echelle de Goldstein",
            min_value=-10.0, max_value=10.0, value=0.0, step=0.1
        )
        st.caption("Negatif = déstabilisant (conflit, violence) · Positif = coopératif (accord, aide)")

        avg_tone = st.slider(
            "Tone Médiatique Moyen",
            min_value=-20.0, max_value=20.0, value=0.0, step=0.1
        )
        st.caption("Sentiment de la presse — Négatif = alarmiste · Positif = favorable")

        mentions = st.slider(
            "Nombre de Mentions",
            min_value=1, max_value=300, value=10, step=1
        )
        st.caption("Citations totales de l'événement dans les flux d'info GDELT")

        sources = st.slider(
            "Nombre de Sources distinctes",
            min_value=1, max_value=50, value=1, step=1
        )
        st.caption("Sites médiatiques indépendants couvrant l'événement")

        articles = st.slider(
            "Nombre d'Articles",
            min_value=1, max_value=300, value=10, step=1
        )
        st.caption("Articles complets publiés — reflète l'ampleur médiatique réelle")

    with sim_col2:
        st.subheader("Résultat du modèle")

        # Prédiction
        features_df = pd.DataFrame([{
            "GoldsteinScale": goldstein,
            "AvgTone":        avg_tone,
            "NumMentions":    mentions,
            "NumSources":     sources,
            "NumArticles":    articles
        }])
        cluster_id   = kmeans_model.predict(scaler_model.transform(features_df))[0]
        cluster_key  = f"Cluster_{cluster_id}"
        cluster_name = cluster_regles[cluster_key]["Nom_Contexte"]
        color        = _CLUSTER_COLORS.get(cluster_name, "#888")
        icon         = _CLUSTER_ICONS.get(cluster_name, "❓")
        ref_vals     = cluster_regles[cluster_key]["Valeurs_Moyennes"]

        # Carte résultat
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border: 2px solid {color}; border-radius: 12px; padding: 24px;
                    box-shadow: 0 0 20px {color}44; text-align: center; margin-bottom: 20px;">
            <div style="font-size: 20px; font-weight: bold; color: {color}; letter-spacing: 3px; margin-bottom: 8px;">{icon}</div>
            <div style="font-size: 11px; color: #888; letter-spacing: 2px; text-transform: uppercase;">Classification ML</div>
            <div style="font-size: 24px; font-weight: bold; color: {color}; margin: 8px 0;">{cluster_name}</div>
            <div style="font-size: 12px; color: #aaa;">Cluster {cluster_id} · Basé sur KMeans</div>
        </div>
        """, unsafe_allow_html=True)

        # Comparaison avec les valeurs de référence
        st.markdown("**Comparaison avec les moyennes du cluster :**")
        compare_data = {
            "Paramètre":        ["Goldstein", "Tone Médiatique", "Mentions", "Sources", "Articles"],
            "Votre scénario":   [goldstein, avg_tone, mentions, sources, articles],
            "Moyenne du cluster": [
                ref_vals["GoldsteinScale"], ref_vals["AvgTone"],
                ref_vals["NumMentions"],   ref_vals["NumSources"], ref_vals["NumArticles"]
            ]
        }
        compare_df = pd.DataFrame(compare_data)
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name="Votre scénario",
            x=compare_df["Paramètre"],
            y=compare_df["Votre scénario"],
            marker_color=color,
            opacity=0.9
        ))
        fig_compare.add_trace(go.Bar(
            name="Moyenne cluster",
            x=compare_df["Paramètre"],
            y=compare_df["Moyenne du cluster"],
            marker_color="#555",
            opacity=0.7
        ))
        fig_compare.update_layout(
            template="plotly_dark",
            barmode="group",
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_compare, use_container_width=True)
