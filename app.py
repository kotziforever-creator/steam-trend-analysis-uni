import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import SteamDataLoader
from src.analytics import SteamAnalytics
import os
import ast

# --- PROJEKT-KONFIGURATION ---
st.set_page_config(
    page_title="Steam Trend Analyse | Baumgart & Schmidt", 
    page_icon="🎮", 
    layout="wide"
)

# --- HILFSFUNKTIONEN ---

def process_tags_safely(tag_data):
    """
    Parsen der Tag-Datenstrukturen mit Fehlerbehandlung.
    
    Da die Rohdaten inkonsistente Formate aufweisen (Teils Dictionary, teils Listen, 
    teils String-Repräsentationen), normalisiert diese Funktion den Input zu einer Liste.
    
    Args:
        tag_data: Rohdaten aus der JSON (dict, list oder str)
    
    Returns:
        list: Eine bereinigte Liste von Tag-Strings.
    """
    if isinstance(tag_data, dict):
        return list(tag_data.keys())
    elif isinstance(tag_data, list):
        return tag_data
    elif isinstance(tag_data, str):
        # Fallback: Versuch, String-Repräsentationen (z.B. aus CSV-Exporten) wiederherzustellen
        try:
            parsed = ast.literal_eval(tag_data)
            if isinstance(parsed, dict): return list(parsed.keys())
            if isinstance(parsed, list): return parsed
            return []
        except (ValueError, SyntaxError):
            return []
    else:
        return []

def main():
    """
    Haupteinstiegspunkt der Streamlit-Applikation.
    Steuert den Control-Flow zwischen Datenladung, Sidebar-Filterung und Tab-Rendering.
    """
    st.title("🎮 Steam Markttrends: Eine datengetriebene Analyse")
    
    st.markdown("""
    **Seminararbeit: Analyse ökonomischer und ludologischer Erfolgsfaktoren**
    
    Dieses Dashboard visualisiert die Ergebnisse unserer Analyse von über 80.000 Steam-Titeln.
    Der Fokus liegt auf der Untersuchung signikanter Korrelationen zwischen Preisgestaltung, 
    Genre-Zugehörigkeit, Nischen-Tags und dem User-Engagement.
    """)

    # Definition der Datenquelle
    DATA_PATH = "data/games.json"
    
    if not os.path.exists(DATA_PATH):
        st.error(f"Runtime Error: Datei '{DATA_PATH}' nicht im Root-Verzeichnis gefunden.")
        return

    # --- ETL-PIPELINE (Cached) ---
    # @st.cache_data optimiert die Performance, indem das Resultat der Datenladung 
    # im Arbeitsspeicher gehalten wird, um Reloads bei Interaktion zu vermeiden.
    @st.cache_data
    def load_and_clean_data():
        loader = SteamDataLoader(DATA_PATH)
        dataframe = loader.prepare_dataframe()
        
        # Sicherstellung der Datenkonsistenz (Schema Validation)
        if 'tags' not in dataframe.columns:
            dataframe['tags'] = [{} for _ in range(len(dataframe))]
        if 'genres' not in dataframe.columns:
            dataframe['genres'] = [[] for _ in range(len(dataframe))]

        # Feature Extraction: Tags normalisieren & Zeitreihen-Features extrahieren
        dataframe['tags_list'] = dataframe['tags'].apply(process_tags_safely)
        dataframe['year'] = dataframe['release_date'].dt.year
        
        return dataframe

    try:
        df = load_and_clean_data()
        
        # --- SIDEBAR: GLOBALE FILTER ---
        st.sidebar.header("📊 Globale Analyse-Parameter")
        st.sidebar.caption("Filter wirken sich auf alle mathematischen Modelle aus.")
        
        # Dynamische Ermittlung der Grenzen für den Slider
        min_year = int(df['year'].min()) if not pd.isna(df['year'].min()) else 2010
        max_year = int(df['year'].max()) if not pd.isna(df['year'].max()) else 2025
        
        if min_year == max_year:
            year_range = (min_year, max_year)
            st.sidebar.warning("Datensatz enthält nur ein Jahr.")
        else:
            year_range = st.sidebar.slider(
                "Betrachteter Release-Zeitraum",
                min_value=min_year, max_value=max_year,
                value=(min_year, max_year)
            )

        # Anwendung der Filtermaske auf den Master-DataFrame
        mask = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
        base_df = df[mask].copy()
        
        st.sidebar.success(f"Datensatz aktiv: {len(base_df)} Titel geladen.")
        
    except Exception as e:
        st.error(f"Pipeline-Fehler: {str(e)}")
        return

    # --- ANALYSE-BEREICHE (Tabs) ---
    tab_price, tab_genre, tab_tags = st.tabs([
        "💰 Monetarisierung", 
        "🎭 Genre-Dynamiken", 
        "🏷️ Nischen-Analyse"
    ])

    # ================= TAB 1: PREIS-ANALYSE =================
    with tab_price:
        st.header("1. Preis-Leistungs-Korrelationen")
        
        # Lokaler Filter für Detailbetrachtung
        price_range = st.slider("Preis-Segment wählen (USD)", 0.0, 100.0, (0.0, 60.0))
        tab1_df = base_df[(base_df['price'] >= price_range[0]) & (base_df['price'] <= price_range[1])]
        
        # KPI Berechnung
        if not tab1_df.empty:
            avg_price = tab1_df['price'].mean()
            # Median ist robuster gegen Ausreißer (z.B. Spiele mit 10k Stunden)
            played_games = tab1_df[tab1_df['average_playtime_forever'] > 0]
            med_playtime = played_games['average_playtime_forever'].median() if not played_games.empty else 0
            avg_score = tab1_df['score_ratio'].mean() * 100
        else:
            avg_price, med_playtime, avg_score = 0, 0, 0

        # Metriken-Darstellung
        k1, k2, k3 = st.columns(3)
        k1.metric("Ø Segment-Preis", f"{avg_price:.2f} $")
        k2.metric("Median Spielzeit", f"{med_playtime:.0f} Min.")
        k3.metric("Segment-Erfolgsrate", f"{avg_score:.1f}%")

        st.divider()
        
        # Visualisierung & Modellierung
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Preisverteilung")
            fig_p_hist = px.histogram(
                tab1_df, x="price", nbins=40, 
                template="plotly_dark",
                color_discrete_sequence=['#636EFA'],
                labels={'price': 'Preis ($)', 'count': 'Anzahl Titel'}
            )
            st.plotly_chart(fig_p_hist, use_container_width=True)
            
        with c2:
            st.subheader("Regressions-Modell (OLS)")
            # Aufruf der statistischen Komponente
            model_summary = SteamAnalytics.perform_linear_regression(tab1_df)
            with st.expander("Mathematischen Report öffnen"):
                st.code(model_summary, language='text')

    # ================= TAB 2: GENRE-ENTWICKLUNG =================
    with tab_genre:
        st.header("2. Zeitliche Entwicklung & Marktsättigung")
        
        if not base_df.empty:
            # Data Expansion: Explode transformiert Listen in Zeilen für granulare Analyse
            genre_exp_df = base_df.explode('genres')
            top_genres = genre_exp_df['genres'].value_counts().nlargest(15).index.tolist()
            
            selected_genre = st.selectbox("Fokus-Genre wählen", top_genres)
            
            # Filterung auf explodierten Daten
            genre_subset = genre_exp_df[genre_exp_df['genres'] == selected_genre]
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.subheader(f"Trendverlauf: {selected_genre}")
                # Aggregation auf Jahresbasis
                genre_stats = genre_subset.groupby('year').agg({
                    'name': 'count',
                    'average_playtime_forever': 'mean'
                }).reset_index()
                
                # Dual-Axis Plot zur Gegenüberstellung von Quantität (Releases) und Qualität (Playtime)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=genre_stats['year'], y=genre_stats['name'], 
                    name="Anzahl Releases", line=dict(color='#00CC96')
                ))
                fig.add_trace(go.Scatter(
                    x=genre_stats['year'], y=genre_stats['average_playtime_forever'], 
                    name="Ø Spielzeit (Min)", yaxis="y2", line=dict(color='#EF553B')
                ))
                
                fig.update_layout(
                    template="plotly_dark",
                    yaxis=dict(title="Releases"),
                    yaxis2=dict(title="Spielzeit", overlaying="y", side="right"),
                    legend=dict(x=0, y=1.2, orientation="h"),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_g2:
                st.subheader("Inferenzstatistik")
                # Für die Regression nutzen wir den Original-DataFrame (ohne Duplikate durch Explode),
                # filtern aber auf Spiele, die das Genre beinhalten.
                genre_orig_df = base_df[base_df['genres'].apply(lambda x: selected_genre in x if isinstance(x, list) else False)]
                
                g_summary = SteamAnalytics.perform_linear_regression(genre_orig_df)
                with st.expander(f"Regressions-Details: {selected_genre}"):
                    st.code(g_summary, language='text')

    # ================= TAB 3: NISCHEN-ANALYSE =================
    with tab_tags:
        st.header("3. Nischen-Analyse (Tags)")
        
        if 'tags_list' in base_df.columns:
            # Pre-Processing: Entfernen leerer Tags
            tag_exp = base_df.explode('tags_list')
            tag_exp = tag_exp[tag_exp['tags_list'].astype(str).str.len() > 1]
            
            top_tags_list = tag_exp['tags_list'].value_counts().nlargest(25).index.tolist()
            
            if top_tags_list:
                sel_tag = st.selectbox("Tag zur Tiefenanalyse wählen", top_tags_list)
                
                # Datensatz filtern
                tag_df = base_df[base_df['tags_list'].apply(lambda x: sel_tag in x if isinstance(x, list) else False)]
                
                c_t1, c_t2 = st.columns([2, 1])
                
                with c_t1:
                    st.subheader(f"Engagement-Matrix: {sel_tag}")
                    
                    # VISUALISIERUNG: Outlier-Bereinigung
                    # Wir filtern extreme Werte NUR für den Plot, um "Verklumpung" zu vermeiden.
                    # Die statistische Analyse (rechts) nutzt weiterhin den vollen Datensatz.
                    plot_df = tag_df[
                        (tag_df['average_playtime_forever'] < 10000) & 
                        (tag_df['price'] <= 120)  # Cut-off für bessere Lesbarkeit
                    ].copy()
                    
                    if not plot_df.empty:
                        fig_bubble = px.scatter(
                            plot_df, 
                            x="price", 
                            y="score_ratio",
                            size="average_playtime_forever",
                            color="average_playtime_forever",
                            hover_name="name",
                            template="plotly_dark",
                            size_max=50,  
                            labels={"score_ratio": "Score", "price": "Preis ($)", "average_playtime_forever": "Zeit"},
                            title=f"Preis vs. Score (Fokus-Ansicht bis 120$)",
                            opacity=0.7 
                        )
                        # Fixierung der X-Achse für sauberes Layout
                        fig_bubble.update_layout(xaxis_range=[-5, 120])
                        st.plotly_chart(fig_bubble, use_container_width=True)
                    else:
                        st.warning("Nicht genügend Datenpunkte für die Visualisierung nach Filterung.")
                
                with c_t2:
                    st.subheader("Statistische Kennzahlen")
                    # Metriken auf Basis der Gesamtpopulation (inkl. Outlier)
                    engagement = tag_df['average_playtime_forever'].mean()
                    st.metric("Engagement Index (Ø Min)", f"{engagement:.0f} Min.")
                    
                    t_stat = SteamAnalytics.perform_linear_regression(tag_df)
                    with st.expander("Modell-Parameter"):
                        st.code(t_stat, language='text')
            else:
                st.info("Keine ausreichenden Tag-Daten im gewählten Zeitraum verfügbar.")
        else:
            st.error("Fehler: Tag-Spalte konnte nicht initialisiert werden.")

if __name__ == "__main__":
    main()