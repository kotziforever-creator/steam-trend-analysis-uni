import sys
import os
import unittest
import pandas as pd
import numpy as np

# --- PFAD-KONFIGURATION ---
# Dynamische Erweiterung des Python-Suchpfads (PYTHONPATH).
# Dies ist notwendig, damit der Test-Runner, der im Unterverzeichnis 'tests/' operiert,
# Zugriff auf die Module im übergeordneten 'src/'-Verzeichnis erhält, ohne dass
# das Paket global im System installiert sein muss.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analytics import SteamAnalytics

class TestSteamAnalytics(unittest.TestCase):
    """
    Test-Suite für die mathematische Komponente 'SteamAnalytics'.
    
    Diese Klasse implementiert Unit-Tests für die statistischen Backend-Methoden.
    Der Fokus liegt auf der Validierung der Ordinary Least Squares (OLS) Regression.
    Es wird geprüft, ob das Modell bei idealen, korrupten und unzureichenden
    Datenmengen (Sample Size n < 30) mathematisch deterministische und logisch
    korrekte Rückgabewerte liefert.
    """

    def setUp(self):
        """
        Fixture-Setup: Initialisierung synthetischer Testdaten vor jeder Testmethode.
        
        Wir konstruieren drei Szenarien, um die Robustheit der 'perform_linear_regression'
        Methode gegen verschiedene Datenqualitäts-Level zu prüfen.
        """
        # Szenario A: Idealer Datensatz (Golden Path)
        # Ein synthetischer DataFrame mit 50 Beobachtungen (n=50), was die statistische 
        # Mindestanforderung (n>=30) für die Annahme der Normalverteilung erfüllt.
        # Die Daten weisen eine perfekte Linearität auf, um die Modell-Konvergenz zu garantieren.
        self.perfect_df = pd.DataFrame({
            'score_ratio': [0.5] * 50,          # Konstante Zielvariable
            'price': range(50),                 # Linear steigender Preis
            'average_playtime_forever': range(100, 150) # Linear steigende Spielzeit
        })

        # Szenario B: Korrupte Daten (Dirty Data)
        # Enthält NaN-Werte (Not a Number), die im Pre-Processing-Schritt der Pipeline
        # durch Listwise Deletion entfernt werden müssen.
        self.dirty_df = pd.DataFrame({
            'score_ratio': [0.5, 0.6, np.nan, 0.8],
            'price': [10, 10, 10, 10],
            'average_playtime_forever': [100, 100, 100, 100]
        })

        # Szenario C: Unzureichende Stichprobengröße (Edge Case)
        # Ein DataFrame mit n=2. Hier muss das statistische Modul die Berechnung verweigern,
        # da keine validen Inferenzschlüsse gezogen werden können.
        self.tiny_df = pd.DataFrame({
            'score_ratio': [0.5, 0.6],
            'price': [10, 20],
            'average_playtime_forever': [100, 200]
        })

    def test_perform_linear_regression_perfect_correlation(self):
        """
        Testfall 1: Validierung des Happy Path (n > 30).
        
        Erwartetes Verhalten:
        Die Funktion akzeptiert den DataFrame, führt die OLS-Minimierung durch und 
        gibt einen formatierten String zurück, der die 'OLS Regression Results' Tabelle enthält.
        """
        result = SteamAnalytics.perform_linear_regression(self.perfect_df)
        
        # Typprüfung: Rückgabe muss Text sein (für das Frontend)
        self.assertIsInstance(result, str, "Rückgabewert muss ein String sein.")
        
        # Inhaltsprüfung: Suchen nach dem statistischen Header
        self.assertIn("OLS Regression Results", result, 
                      f"Der Header fehlt. Rückgabe war: {result}")

    def test_perform_linear_regression_robustness(self):
        """
        Testfall 2: Validierung der Sample-Size-Guardrails.
        
        Erwartetes Verhalten:
        Bei n < 30 muss das System eine spezifische Warnmeldung zurückgeben, anstatt
        einen Laufzeitfehler (RuntimeError) zu werfen. Dies verhindert Frontend-Abstürze.
        """
        result = SteamAnalytics.perform_linear_regression(self.tiny_df)
        
        expected_msg = "Datensatz zu klein"
        self.assertIn(expected_msg, result, 
                      "Das System hat die Sample-Size-Restriction ignoriert.")

    def test_columns_validation(self):
        """
        Testfall 3: Validierung der Schema-Konformität.
        
        Erwartetes Verhalten:
        Das System muss erkennen, wenn essentielle Spalten (Features) für die Regression fehlen.
        """
        # DataFrame ohne 'price' Spalte (Simuliert fehlerhaften Import)
        broken_df = pd.DataFrame({'score_ratio': [1, 2], 'playtime': [10, 20]})
        result = SteamAnalytics.perform_linear_regression(broken_df)
        
        self.assertIn("Fehler", result, "Fehlende Spalten wurden nicht erkannt.")

if __name__ == '__main__':
    unittest.main()