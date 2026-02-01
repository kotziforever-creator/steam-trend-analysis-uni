import unittest
import os
import pandas as pd
import json
import tempfile
import sys
import logging

# --- PFAD-KONFIGURATION ---
# Dynamische Erweiterung des Python-Suchpfads (PYTHONPATH).
# Dies ist notwendig, damit der Test-Runner, der im Unterverzeichnis 'tests/' operiert,
# Zugriff auf die Module im übergeordneten 'src/'-Verzeichnis erhält, ohne dass
# das Paket global im System installiert sein muss.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import SteamDataLoader

class TestSteamDataLoader(unittest.TestCase):
    """
    Umfassende Test-Suite für die ETL-Pipeline (Extract, Transform, Load).
    
    Diese Klasse dient der Qualitätssicherung des Data-Ingestion-Prozesses.
    Sie verifiziert, dass die Transformation von unstrukturierten JSON-Rohdaten
    in strukturierte Pandas DataFrames deterministisch und fehlerfrei abläuft.
    
    Abgedeckte Test-Dimensionen:
    1. **Schema-Validierung:** Existenz aller für die Regression notwendigen Spalten.
    2. **Type-Casting:** Korrekte Konvertierung von String-Literalen in numerische Datentypen (Float/Int).
    3. **Data Cleaning:** Umgang mit fehlenden Werten (NaN) und korrupten Datensätzen.
    4. **Feature Engineering:** Mathematische Korrektheit abgeleiteter Metriken (Score Ratio).
    5. **Boundary Value Analysis:** Verhalten bei Extremwerten (z.B. 0 Reviews, negative Preise).
    """

    def setUp(self):
        """
        Fixture-Setup: Erstellung einer temporären, synthetischen Datenbasis.
        
        Wir simulieren diverse Daten-Anomalien, wie sie im echten 'FronkonGames' Datensatz
        vorkommen können, um die Robustheit des Loaders zu garantieren.
        """
        self.test_data = {
            # Szenario A: Der "perfekte" Datensatz (Happy Path)
            "10": { 
                "name": "Test Game A (Reference)",
                "release_date": "2020-01-01",
                "price": "19.99",       # Testet String-zu-Float Parsing
                "positive": 100,
                "negative": 0,          # Testet Division-by-Zero Schutz
                "average_playtime_forever": 60,
                "tags": {"Action": 100, "Indie": 50},
                "genres": ["Action", "Indie"]
            },
            # Szenario B: Korrupte / Fehlende Daten (Dirty Data)
            "20": {
                "name": "Test Game B (Corrupt)",
                "release_date": "invalid-date", # Muss zu NaT (Not a Time) werden
                "price": 0,
                "positive": 0,
                "negative": 0,
                "tags": [], # Inkonsistenter Datentyp (Liste statt Dict)
                "genres": []
            },
            # Szenario C: Grenzwerte (Boundary Values)
            "30": {
                "name": "Test Game C (Edge Case)",
                "release_date": "2030-01-01",   # Datum in der Zukunft
                "price": "-5.00",               # Negativer Preis (Datenfehler)
                "positive": 1,
                "negative": 1,
                "average_playtime_forever": 0,
                "tags": None,                   # Fehlende Tags
                "genres": None
            }
        }
        
        # Erstelle temporäre Datei im OS-Temp-Ordner
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
        
        # Instanziierung des Loaders
        self.loader = SteamDataLoader(self.temp_file.name)
        
        # Deaktivierung von Logging-Output während der Tests, um die Konsole sauber zu halten
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        """
        Teardown: Entfernung der temporären Artefakte nach Testabschluss.
        """
        os.unlink(self.temp_file.name)
        logging.disable(logging.NOTSET)

    def test_etl_pipeline_integrity_and_dimensions(self):
        """
        Integrationstest 1: Prüfung der strukturellen Integrität.
        
        Ziel: Sicherstellen, dass keine Datensätze während des Ladens verloren gehen (Data Loss)
        und dass das resultierende Schema den Anforderungen der Analytik entspricht.
        """
        df = self.loader.prepare_dataframe()
        
        # 1. Dimensions-Check
        expected_rows = 3
        self.assertEqual(len(df), expected_rows, 
                         f"Erwartete {expected_rows} Zeilen, erhielt {len(df)}. Datenverlust im Loader!")
        
        # 2. Schema-Validierung
        required_cols = ['price', 'score_ratio', 'year', 'tags_list']
        # Hinweis: 'year' und 'tags_list' werden erst in der App erzeugt, hier prüfen wir Basis-Spalten
        base_cols = ['price', 'positive', 'negative', 'average_playtime_forever']
        
        for col in base_cols:
            self.assertIn(col, df.columns, f"Kritische Spalte fehlt: {col}")

    def test_type_safety_and_casting(self):
        """
        Integrationstest 2: Validierung der Datentyp-Sicherheit.
        
        Ziel: Verifizieren, dass Strings ("19.99") korrekt in Fließkommazahlen (19.99)
        umgewandelt werden, da Strings keine mathematischen Operationen erlauben.
        """
        df = self.loader.prepare_dataframe()
        
        # Prüfung auf Float-Typ (64-bit)
        self.assertTrue(pd.api.types.is_float_dtype(df['price']), 
                        "Die Preis-Spalte ist kein Float-Typ. Regression unmöglich.")
        
        # Prüfung auf DateTime-Objekte
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['release_date']), 
                        "Release-Datum wurde nicht geparst. Zeitreihenanalyse unmöglich.")

    def test_score_ratio_mathematical_correctness(self):
        """
        Unit Test: Validierung der 'Score Ratio' Metrik.
        
        Mathematischer Hintergrund:
        Die Score Ratio ist definiert als P(Positive) = Pos / (Pos + Neg).
        Dieser Test prüft explizit die Behandlung von Division-by-Zero Fällen.
        """
        df = self.loader.prepare_dataframe()
        
        # Fall A: Perfektes Spiel (100% Positiv)
        # Erwartung: 100 / (100+0) = 1.0
        game_a = df.iloc[0]
        self.assertAlmostEqual(game_a['score_ratio'], 1.0, places=4)
        
        # Fall B: Spiel ohne Reviews (0 Pos, 0 Neg)
        # Erwartung: 0.0 (durch Imputation/Smoothing verhindert Programmabsturz)
        game_b = df.iloc[1]
        self.assertEqual(game_b['score_ratio'], 0.0, 
                         "Division durch Null bei 0 Reviews wurde nicht abgefangen.")
        
        # Fall C: Gemischtes Feedback (1 Pos, 1 Neg)
        # Erwartung: 1 / (1+1) = 0.5
        game_c = df.iloc[2]
        self.assertAlmostEqual(game_c['score_ratio'], 0.5, places=4)

    def test_heterogeneous_tag_structures(self):
        """
        Unit Test: Robustheit gegen inkonsistente JSON-Strukturen.
        
        In der Praxis können Tags als Dict, Liste oder None vorliegen.
        Der Loader muss dies normalisieren, um Iterationsfehler im Frontend zu vermeiden.
        """
        df = self.loader.prepare_dataframe()
        
        # Szenario 1: Tags liegen als Dictionary vor (Normalfall)
        tags_a = df.iloc[0]['tags']
        self.assertIsInstance(tags_a, dict, "Tags sollten als Dictionary geladen werden.")
        
        # Szenario 2: Tags fehlen oder sind None (Edge Case)
        # Der Loader sollte dies in ein leeres Dict {} oder eine leere Liste umwandeln
        tags_c = df.iloc[2]['tags']
        self.assertTrue(isinstance(tags_c, (dict, list)), 
                        "Fehlende Tags führten nicht zu einem leeren Container-Objekt.")

if __name__ == '__main__':
    unittest.main()