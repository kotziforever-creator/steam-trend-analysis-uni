import pandas as pd
import os
import numpy as np
import logging

# Logger-Konfiguration für Nachvollziehbarkeit der ETL-Schritte
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SteamDataLoader:
    """
    Implementiert eine robuste ETL-Pipeline (Extract, Transform, Load) für 
    heterogene Steam-Datensätze.
    
    Diese Komponente verantwortet die Ingestierung unstrukturierter Rohdaten (JSON),
    die Validierung der Datenintegrität sowie die Transformation in ein relationales 
    DataFrame-Format (Pandas), das für vektorisierte mathematische Operationen 
    (z.B. OLS-Regression) optimiert ist.
    
    Attributes:
        file_path (str): Der absolute oder relative Pfad zur Quelldatei.
        df (pd.DataFrame): Der volatile Speicher für den transformierten Datensatz.
    """

    def __init__(self, file_path):
        """
        Initialisiert den DataLoader.
        
        Args:
            file_path (str): Systempfad zur 'games.json' Datei.
        """
        self.file_path = file_path
        self.df = None

    def _validate_source(self):
        """
        Interner Validierungs-Check ("Fail-Fast"), um I/O-Fehler vor der 
        speicherintensiven Verarbeitung abzufangen.
        """
        if not os.path.exists(self.file_path):
            error_msg = f"ETL-Error: Quelldatei {self.file_path} nicht gefunden."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        return True

    def prepare_dataframe(self):
        """
        Führt den primären Transformationsprozess durch.
        
        Ablauf der Pipeline:
        1. Ingestion: Laden der JSON-Struktur (optimiert für große Dateien).
        2. Projection: Reduktion auf statistisch relevante Feature-Spalten.
        3. Imputation: Behandlung fehlender Werte (NaN) bei Preisen und Metriken.
        4. Feature Engineering: Berechnung abgeleiteter KPIs (z.B. Score Ratio).
        
        Returns:
            pd.DataFrame: Der bereinigte Datensatz, bereit für die Analyse.
        """
        self._validate_source()
        
        logger.info(f"Starte Daten-Ingestierung von: {self.file_path}")

        # 1. INGESTION
        # Wir nutzen 'orient=index', da Steam-Daten oft als Hash-Map (Key=AppID) vorliegen.
        # Fallback-Mechanismus greift, falls das JSON als Array strukturiert ist.
        try:
            temp_df = pd.read_json(self.file_path, orient='index')
        except ValueError:
            logger.warning("JSON-Struktur weicht ab (Array statt Index). Nutze Fallback-Parser.")
            temp_df = pd.read_json(self.file_path)

        # 2. PROJECTION (Feature Selection)
        # Definition der für die Hypothesenprüfung relevanten Dimensionen.
        # 'tags' und 'genres' sind essentiell für die Cluster-Analysen.
        important_columns = [
            'name', 
            'release_date', 
            'positive', 
            'negative', 
            'price', 
            'average_playtime_forever', 
            'genres', 
            'tags'
        ]
        
        # Schnittmenge bilden, um KeyErrors bei unvollständigen Datensätzen zu vermeiden
        existing_cols = [c for c in important_columns if c in temp_df.columns]
        self.df = temp_df[existing_cols].copy()
        
        # 3. DATA CLEANING & TYPE CASTING
        
        # Zeitreihen-Konvertierung
        self.df['release_date'] = pd.to_datetime(self.df['release_date'], errors='coerce')
        
        # Numerische Stabilisierung (Imputation von 0 für fehlende Werte)
        cols_to_numeric = ['price', 'positive', 'negative', 'average_playtime_forever']
        for col in cols_to_numeric:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # Handling semi-strukturierter Daten (Nested JSON in Tags/Genres)
        # Wir erzwingen den Datentyp 'dict' für Tags, um Iterationsfehler im Frontend zu vermeiden.
        if 'tags' in self.df.columns:
            self.df['tags'] = self.df['tags'].apply(
                lambda x: x if isinstance(x, (dict, list)) else {}
            )
        else:
            self.df['tags'] = [{} for _ in range(len(self.df))]

        if 'genres' not in self.df.columns:
             self.df['genres'] = [[] for _ in range(len(self.df))]

        # 4. FEATURE ENGINEERING (Mathematische Modellierung)
        
        # Berechnung der 'Score Ratio' als Proxy für Kundenzufriedenheit.
        # Formel: R = Pos / (Pos + Neg)
        # Division-by-Zero Protection: Nenner wird mindestens auf 1 gesetzt.
        self.df['total_reviews'] = self.df['positive'] + self.df['negative']
        self.df['score_ratio'] = self.df['positive'] / self.df['total_reviews'].replace(0, 1)
        
        logger.info(f"ETL erfolgreich abgeschlossen. Datensatzgröße: {self.df.shape}")
        
        return self.df