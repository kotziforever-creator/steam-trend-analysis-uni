import os
import shutil
import logging
from pathlib import Path

# Wir versuchen kagglehub zu importieren, fangen aber Fehler ab, 
# falls die Library in der Python-Umgebung fehlt.
try:
    import kagglehub
except ImportError:
    kagglehub = None

# --- LOGGING KONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class KaggleDataIngestion:
    """
    Verwaltet den automatisierten Download und die Bereitstellung von Datensätzen 
    über die offizielle Kaggle-API.
    
    Diese Klasse kapselt die Authentifizierungslogik und das Dateisystem-Management,
    um sicherzustellen, dass die Rohdaten (JSON) an einem deterministischen Ort 
    für den 'SteamDataLoader' zur Verfügung stehen.
    """

    def __init__(self):
        """
        Initialisiert die Umgebungsvariablen für die API-Authentifizierung.
        
        HINWEIS ZUR SICHERHEIT:
        In einer Produktionsumgebung würden diese Credentials selbstverständlich via Environment-Variables 
        oder Vault-Services injiziert werden. Für den Kontext dieser Seminararbeit 
        sind sie hardcodiert, um die direkte Ausführbarkeit (Reproduzierbarkeit) für den Prüfer zu gewährleisten. Es handelt sich um einen Test Account!
        """
        self.dataset_handle = "fronkongames/steam-games-dataset"
        
        # Authentifizierungs-Token für den API-Handshake
        os.environ["KAGGLE_USERNAME"] = "constimisscoololo" 
        os.environ["KAGGLE_KEY"] = "KGAT_d6193e7b17b6ea04d0464632e55500d8"

    def fetch_and_place_data(self):
        """
        Führt den Download durch und migriert die Daten in die Projektstruktur.
        
        Ablauf:
        1. API-Call via kagglehub (nutzt Caching, um Bandbreite zu sparen).
        2. Identifikation des Projekt-Root-Verzeichnisses.
        3. Verschieben der relevanten JSON-Datei in den 'data/'-Ordner.
        """
        if kagglehub is None:
            logger.error("Modul 'kagglehub' fehlt. Bitte `pip install kagglehub` ausführen.")
            return

        logger.info(f"Initialisiere API-Download für: {self.dataset_handle}")
        
        try:
            # 1. Download in den lokalen Cache (Appdata/Local/...)
            path = kagglehub.dataset_download(self.dataset_handle)
            logger.info(f"Rohdaten erfolgreich im Cache zwischengespeichert: {path}")

            # 2. Pfad-Berechnung
            # Wir gehen davon aus, dass dieses Skript in /src/ liegt. 
            # Parent -> Parent ist das Root-Verzeichnis.
            current_script_path = Path(__file__).resolve()
            project_root = current_script_path.parent.parent
            target_folder = project_root / "data"
            
            # Ordnerstruktur erzwingen (idempotent)
            if not target_folder.exists():
                logger.info(f"Erstelle Zielverzeichnis: {target_folder}")
                target_folder.mkdir(parents=True, exist_ok=True)

            # 3. Datei-Migration
            file_moved = False
            for entry in os.scandir(path):
                if entry.name.endswith(".json") and entry.is_file():
                    source_path = entry.path
                    target_path = target_folder / "games.json"
                    
                    # Kopiervorgang
                    shutil.copy2(source_path, target_path)
                    
                    logger.info(f"ETL-Abschluss: Datei bereitgestellt unter: {target_path}")
                    file_moved = True
                    break
            
            if not file_moved:
                logger.warning("Keine JSON-Datei im heruntergeladenen Paket gefunden.")

        except Exception as e:
            logger.critical(f"Kritischer Fehler bei der Datenbeschaffung: {str(e)}")
            raise e

if __name__ == "__main__":
    # Instanziierung und Ausführung
    ingestor = KaggleDataIngestion()
    ingestor.fetch_and_place_data()