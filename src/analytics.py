import pandas as pd
import statsmodels.api as sm
import logging

# Logger für Nachvollziehbarkeit der statistischen Berechnungen
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SteamAnalytics:
    """
    Stellt die mathematische Modellierungsebene der Applikation bereit.
    
    Diese Klasse kapselt komplexe statistische Verfahren, insbesondere aus dem Bereich
    der Inferenzstatistik, um kausale Zusammenhänge in den Marktdaten zu quantifizieren.
    Der primäre Fokus liegt auf linearen Regressionsmodellen (OLS), die es ermöglichen,
    den Einfluss unabhängiger Variablen (Preis, Spielzeit) auf eine abhängige 
    Zielvariable (User-Bewertung) zu isolieren.
    """
    
    @staticmethod
    def perform_linear_regression(df):
        """
        Berechnet eine multiple lineare Regression nach der Methode der kleinsten Quadrate
        (Ordinary Least Squares - OLS).
        
        Mathematischer Hintergrund:
        -------------------------
        Das Modell postuliert einen linearen Zusammenhang der Form:
        
            Y = β₀ + β₁X₁ + β₂X₂ + ... + ε
            
        Hierbei repräsentieren:
        - Y (Endogene Variable): Die 'score_ratio' (Erfolgswahrscheinlichkeit 0.0 - 1.0).
        - X (Exogene Variablen): Der Vektor der Prädiktoren ('price', 'average_playtime_forever').
        - β (Koeffizienten): Die zu schätzenden Parameter, die die Stärke und Richtung 
          des Effekts angeben (Steigung der Regressionsgeraden im n-dimensionalen Raum).
        - ε (Residuum): Der Fehlerterm, der die nicht erklärte Varianz des Modells enthält.
        
        Optimierungsverfahren:
        Das Modell minimiert die Summe der quadrierten Residuen (Residual Sum of Squares, RSS):
        
            min Σ (y_i - ŷ_i)²
            
        Dies liefert unter den Gauß-Markow-Annahmen die besten linearen erwartungstreuen 
        Schätzer (BLUE) für die Koeffizienten.
        
        Args:
            df (pd.DataFrame): Der Eingabe-Datensatz. Muss zwingend die Spalten 
                               'score_ratio', 'price' und 'average_playtime_forever' enthalten.
        
        Returns:
            str: Eine formatierte Zusammenfassung der Regressionsstatistiken (Summary Table).
                 Enthält R-Squared, F-Statistik, p-Werte und Konfidenzintervalle.
                 Gibt eine Fehlermeldung zurück, falls die Datenbasis unzureichend ist.
        """
        # 1. Daten-Validierung & Preprocessing
        required_cols = ['score_ratio', 'price', 'average_playtime_forever']
        
        # Prüfung auf Schema-Konformität
        if not all(col in df.columns for col in required_cols):
            error_msg = "Fehler: Datensatz fehlen erforderliche Spalten für die Regression."
            logger.error(error_msg)
            return error_msg
            
        # Bereinigung: Regressionen können mathematisch nicht mit NaN (Not a Number) oder Inf umgehen.
        # Wir entfernen betroffene Zeilen (Listwise Deletion).
        model_df = df[required_cols].dropna()
        
        # Statistische Signifikanz-Prüfung (Sample Size Check)
        # Faustregel: Mindestens 30 Beobachtungen für halbwegs valide Normalverteilungsannahme (Zentraler Grenzwertsatz).
        if len(model_df) < 30:
            logger.warning(f"Regression abgebrochen: Zu wenige Datenpunkte ({len(model_df)}).")
            return "Datensatz zu klein für eine statistisch valide Inferenzanalyse (n < 30)."
        
        try:
            # 2. Definition der Variablen
            # Endogene Variable (Zielgröße)
            y = model_df['score_ratio']
            
            # Exogene Variablen (Prädiktoren)
            X = model_df[['price', 'average_playtime_forever']]
            
            # 3. Modell-Spezifikation
            # Hinzufügen einer Konstanten (Intercept/β₀).
            # Ohne diesen Schritt würde die Regressionsgerade zwangsweise durch den Ursprung (0,0) gehen,
            # was eine starke Verzerrung (Bias) zur Folge hätte, da auch kostenlose Spiele 
            # nicht zwingend eine Bewertung von 0.0 haben.
            X = sm.add_constant(X)
            
            # 4. Modell-Fitting
            # Berechnung der Parameter durch Lösung der Normalgleichung.
            model = sm.OLS(y, X).fit()
            
            logger.info(f"Regressionsmodell erfolgreich berechnet. R²: {model.rsquared:.4f}")
            
            # 5. Rückgabe der Ergebnisse
            # Wir nutzen .as_text(), um die formatierte Tabelle direkt im Dashboard rendern zu können.
            return model.summary().as_text()
            
        except Exception as e:
            logger.error(f"Mathematischer Fehler während der Regression: {str(e)}")
            return f"Berechnungsfehler im OLS-Modell: {str(e)}"