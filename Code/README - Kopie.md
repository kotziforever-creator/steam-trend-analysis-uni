# Ökonomische und Ludologische Erfolgsfaktoren auf Steam: Eine empirische Analyse

## 1. Projektkontext und Forschungsziel
Dieses Softwareprojekt wurde im Rahmen der Seminararbeit im Wintersemester 2025/2026 konzipiert. Das primäre Forschungsziel besteht in der empirischen Quantifizierung von Erfolgsfaktoren auf digitalen Distributionsplattformen, spezifisch Valve's Steam-Plattform.

In einer Zeit, in der täglich Dutzende neue Titel veröffentlicht werden ("Indiepocalypse"), reicht das reine "Bauchgefühl" zur Marktanalyse nicht mehr aus. Dieses Projekt verwendet daher einen **datengetriebenen Ansatz (Data Science)**, um folgende Kernfragen mathematisch zu beantworten:

* **Monetarisierung:** Korreliert ein höherer Verkaufspreis ($) mit einer besseren Nutzerbewertung (Score Ratio)?
* **Engagement:** Ist die Spielzeit ein valider Prädiktor für die Qualität eines Titels?
* **Marktdynamik:** Wie haben sich Nischen-Genres ("Tags") über die letzte Dekade entwickelt?

Die Analyse basiert auf einem Datensatz von über **80.000 Videospielen** und nutzt inferenzstatistische Methoden, um Hypothesen zu validieren oder zu falsifizieren.

---

## 2. Technische Architektur

Das Projekt folgt einer strikten Trennung von Datenverarbeitung (Backend/ETL), statistischer Analyse und Visualisierung (Frontend).

### 2.1 Systemvoraussetzungen
Zur Reproduktion der Ergebnisse wird folgende Umgebung benötigt:
* **Python Runtime:** Version 3.9 oder höher.
* **Paketverwaltung:** pip (empfohlen in einer virtuellen Umgebung `venv`).

### 2.2 Installation & Setup
Befolgen Sie diese Schritte, um die Analyse auf einem lokalen System auszuführen:

1.  **Abhängigkeiten installieren:**
    Laden Sie die notwendigen Bibliotheken für Data Science und Web-Rendering:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Daten-Ingestierung (Automatisierter Download):**
    Das Skript `src/kaggle_import.py` automatisiert den Download des Datensatzes über die Kaggle-API und platziert ihn in der korrekten Ordnerstruktur.
    ```bash
    python src/kaggle_import.py
    ```

3.  **Start der Applikation:**
    Starten Sie das Dashboard über Streamlit:
    ```bash
    streamlit run app.py
    ```

---

## 3. Methodik der Datenverarbeitung (ETL-Pipeline)

Um die Rohdaten (JSON) für die mathematische Analyse nutzbar zu machen, wurde eine robuste **ETL-Pipeline (Extract, Transform, Load)** in der Klasse `SteamDataLoader` (`src/data_loader.py`) implementiert.

### 3.1 Extract (Datenakquise)
Die Daten stammen aus dem *Steam Games Dataset* (FronkonGames). Da die JSON-Datei sehr groß ist und komplexe, verschachtelte Strukturen aufweist, wird sie mittels eines optimierten Parsers (`pd.read_json` mit `orient='index'`) in den Arbeitsspeicher geladen.

### 3.2 Transform (Bereinigung & Feature Engineering)
Hier findet die mathematische Vorverarbeitung statt:
* **Imputation:** Fehlende Werte (NaN) bei Preisen oder Spielzeiten werden durch 0 ersetzt, um die statistischen Modelle nicht zu verzerren (Bias-Vermeidung).
* **Feature Extraction:** Verschachtelte JSON-Objekte in den Spalten `tags` und `genres` werden "explodiert" (unnested) und normalisiert.
* **Score-Berechnung:** Aus den absoluten Zahlen von `positive` und `negative` Reviews wird die relative **Score Ratio** berechnet:
    $$ \text{Score Ratio} = \frac{\text{Positive Reviews}}{\text{Total Reviews}} $$
    Dies normalisiert den Erfolgswert auf eine Skala von 0.0 bis 1.0.

---

## 4. Mathematische Modelle und Statistik

Das Herzstück der Analyse bildet die Klasse `SteamAnalytics` (`src/analytics.py`), die Methoden der **Inferenzstatistik** anwendet.

### 4.1 Multiple Lineare Regression (OLS)
Wir nutzen die **Ordinary Least Squares (OLS)** Methode, um den Einfluss unabhängiger Variablen auf die abhängige Variable (Bewertung) zu isolieren. Das Modell folgt der Gleichung:

$$ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \epsilon $$

Dabei gilt:
* $Y$: Die zu erklärende Variable (Score Ratio).
* $X_1$: Der Preis des Spiels (in USD).
* $X_2$: Die durchschnittliche Spielzeit (Engagement).
* $\beta_i$: Die Regressionskoeffizienten (Steigung der Geraden).
* $\epsilon$: Der Fehlerterm (Residuum).

### 4.2 Interpretation der Kennzahlen
Das Dashboard gibt folgende statistische Kennzahlen aus:
* **R-Squared ($R^2$):** Gibt an, wie viel Prozent der Varianz der Bewertungen durch unser Modell erklärt werden kann.
* **P-Wert ($P>|t|$):** Dient zur Überprüfung der Signifikanz. Werte $< 0.05$ deuten darauf hin, dass der beobachtete Zusammenhang statistisch signifikant ist.
* **Koeffizienten:** Zeigen die Wirkungsrichtung an. Ein negativer Preis-Koeffizient bedeutet beispielsweise: "Je teurer das Spiel, desto kritischer die Bewertung."

---

## 5. Projektstruktur

Die Codebasis ist modular aufgebaut, um Wartbarkeit und Erweiterbarkeit zu gewährleisten:

```text
├── app.py                  # Frontend: Streamlit Dashboard & UI-Logik
├── requirements.txt        # Liste der Python-Abhängigkeiten
├── README.md               # Projektdokumentation
├── data/                   # Speicherort für games.json (lokal generiert)
└── src/                    # Backend-Logik
    ├── __init__.py
    ├── analytics.py        # Statistik-Modul (OLS Regression)
    ├── data_loader.py      # ETL-Pipeline & Data Cleaning
    └── kaggle_import.py    # API-Schnittstelle zu Kaggle