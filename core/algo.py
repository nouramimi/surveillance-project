import pandas as pd
import numpy as np
import os
from datetime import datetime

# On récupère le chemin du dossier où se trouve algo.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fichier d'entrée
fichier_surveillance = os.path.join(BASE_DIR, "surveillanc1.xlsx")


# Fonction principale
def gestion_surveillance():
    # Chemin du fichier Excel (à modifier selon votre configuration)
    #fichier_surveillance = "surveillanc1.xlsx"
    
    # 1. Calcul de la charge de surveillance pour chaque enseignant
    df_enseignant = calcul_charge_supposee(fichier_surveillance)
    
    # 2. Jointure des feuilles seances et filieresalle et calcul du nombre de séances par salle
    nbr_seance_salle, df_seances_salles = calcul_seances_salles(fichier_surveillance)
    
    # 3. Calcul de la charge totale
    charge_totale = df_enseignant['charge supposee'].sum()
    print(f"Charge totale de tous les enseignants: {charge_totale}")
    
    # 4. Calcul du nombre de séances obligatoires pour chaque enseignant
    df_enseignant = calcul_charge_obligatoire(df_enseignant, nbr_seance_salle, charge_totale)
    
    # 5. Calcul du nombre de séances à surveiller par créneau (date, horaire)
    df_seances_par_creneau = calcul_seances_par_creneau(df_seances_salles)
    
    # 6. Jointure des feuilles enseignant et seances par matière
    df_enseignant_seances = jointure_enseignant_seances(fichier_surveillance)
    
    # Enregistrement des résultats
    sauvegarder_resultats(df_enseignant, df_seances_salles, df_seances_par_creneau, df_enseignant_seances)
    
    return df_enseignant, df_seances_salles, df_seances_par_creneau, df_enseignant_seances

# 1. Calcul de la charge supposée pour chaque enseignant
def calcul_charge_supposee(fichier):
    df_enseignant = pd.read_excel(fichier, sheet_name='enseignant')
    
    # Formule: (1*Cours + 1.25*TD + 0.75*TP) * coef
    df_enseignant['charge supposee'] = (df_enseignant['Cours'] + 
                                        1.25 * df_enseignant['TD'] + 
                                        0.75 * df_enseignant['TP']) * df_enseignant['coef']
    
    print("Calcul de la charge supposée terminé")
    return df_enseignant

# 2. Jointure des feuilles seances et filieresalle et calcul du nombre de séances par salle
def calcul_seances_salles(fichier):
    df_seances = pd.read_excel(fichier, sheet_name='seances')
    df_filieresalle = pd.read_excel(fichier, sheet_name='filieresalle')
    
    # Jointure sur la colonne filiere
    df_seances_salles = pd.merge(df_seances, df_filieresalle, on='filiere', how='inner')
    
    # Compter le nombre de combinaisons uniques (date, horaire, salle)
    nbr_seance_salle = (df_seances_salles.drop_duplicates(subset=['date', 'horaire', 'salle']).shape[0])*2
    print(f"Nombre de séances par salle uniques: {nbr_seance_salle}")
    
    return nbr_seance_salle, df_seances_salles

# 4. Calcul du nombre de séances obligatoires pour chaque enseignant
def calcul_charge_obligatoire(df_enseignant, nbr_seance_salle, charge_totale):
    df_enseignant['charge obligatoire'] = np.round(
        df_enseignant['charge supposee'] * (nbr_seance_salle / charge_totale)
    )
    
    print("Calcul de la charge obligatoire terminé")
    return df_enseignant

# 5. Calcul du nombre de séances à surveiller par créneau (date, horaire)
def calcul_seances_par_creneau(df_seances_salles):
    # Grouper par date et horaire, compter les salles uniques
    df_seances_par_creneau = df_seances_salles.groupby(['date', 'horaire'])['salle'].nunique().reset_index()
    df_seances_par_creneau.rename(columns={'salle': 'nombre_salles'}, inplace=True)
    
    print("Calcul du nombre de séances par créneau terminé")
    return df_seances_par_creneau

# 6. Jointure des feuilles enseignant et seances par matière
def jointure_enseignant_seances(fichier):
    df_enseignant = pd.read_excel(fichier, sheet_name='enseignant')
    df_seances = pd.read_excel(fichier, sheet_name='seances')
    
    # Supposons que 'matiere' dans la feuille enseignant correspond à 'matiere' dans la feuille seances
    # Si ce n'est pas le cas, ajustez le nom des colonnes en conséquence
    df_enseignant_seances = pd.merge(
        df_seances[['date', 'horaire', 'matiere']], 
        df_enseignant[['Nom Et Prénom Enseignant', 'matiere']], 
        on='matiere', 
        how='inner'
    )
    
    # Renommer la colonne pour plus de clarté
    df_enseignant_seances.rename(columns={'Nom Et Prénom Enseignant': 'enseignant'}, inplace=True)
    
    print("Jointure des enseignants et séances par matière terminée")
    return df_enseignant_seances

# Fonction pour sauvegarder les résultats
def sauvegarder_resultats(df_enseignant, df_seances_salles, df_seances_par_creneau, df_enseignant_seances):
    # Créer un dossier pour les résultats s'il n'existe pas
    dossier_resultats = "resultats_surveillance"
    if not os.path.exists(dossier_resultats):
        os.makedirs(dossier_resultats)
    
    # Format de date pour les noms de fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder le fichier des enseignants avec les charges calculées
    df_enseignant.to_excel(f"{dossier_resultats}/enseignants_charges.xlsx", index=False)
    
    # Sauvegarder le fichier des séances avec salles
    df_seances_salles.to_excel(f"{dossier_resultats}/seances_salles.xlsx", index=False)
    
    # Sauvegarder le fichier des séances par créneau
    df_seances_par_creneau.to_excel(f"{dossier_resultats}/seances_par_creneau.xlsx", index=False)
    
    # Sauvegarder le fichier des enseignants et séances par matière
    df_enseignant_seances.to_excel(f"{dossier_resultats}/enseignant_seances.xlsx", index=False)
    
    print(f"Tous les fichiers ont été sauvegardés dans le dossier '{dossier_resultats}'")

# Exécuter le programme si le script est lancé directement
if __name__ == "__main__":
    print("Démarrage du programme de gestion des surveillances d'examens...")
    gestion_surveillance()
    print("Programme terminé avec succès.")