from django.shortcuts import render
import subprocess
import os
import pandas as pd
from django.http import FileResponse, HttpResponse, JsonResponse
from django.conf import settings

def run_algos_and_download(request):
    try:
        subprocess.run(["python", "core/algo.py"], check=True)
        subprocess.run(["python", "core/algo1.py"], check=True)

        file_path = os.path.join(settings.BASE_DIR, "planning_surveillance_optimise.xlsx")
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), as_attachment=True, filename="planning_surveillance_optimise.xlsx")
        return HttpResponse("Fichier non trouvé", status=404)
    except subprocess.CalledProcessError as e:
        return HttpResponse(f"Erreur d'exécution : {e}", status=500)

def run_algos(request):
    return HttpResponse("Algorithmes exécutés avec succès.")


def get_teacher_schedule(request, nom_enseignant):
    import traceback
    import os
    import pandas as pd
    from django.http import JsonResponse
    from django.conf import settings

    try:
        file_path = os.path.join(settings.BASE_DIR, 'core', 'resultats_surveillance', 'planning_surveillance_optimise.xlsx')

        # Lire avec multi-index (2 lignes d'en-têtes)
        df = pd.read_excel(file_path, header=[0, 1])
        print("Colonnes multi-index :", df.columns.tolist())

        # Identifier dynamiquement la colonne contenant les enseignants
        enseignant_col = df.columns[0]  # ex : ('Enseignant', '')
        nom_enseignant = nom_enseignant.strip()

        # Nettoyer les noms d'enseignants
        df[enseignant_col] = df[enseignant_col].astype(str).str.strip()

        # Rechercher la ligne
        row = df[df[enseignant_col] == nom_enseignant]
        if row.empty:
            return JsonResponse({'error': f'Enseignant "{nom_enseignant}" non trouvé'}, status=404)

        # Extraire planning
        schedule = []
        for col in df.columns[1:]:  # ignorer la colonne enseignant
            value = row.iloc[0][col]
            if pd.notna(value):
                date_str = str(col[0]).strip()
                hour_str = str(col[1]).strip()
                schedule.append({
                    'date': date_str,
                    'heure': hour_str,
                    'surveillance': value
                })

        return JsonResponse({'enseignant': nom_enseignant, 'planning': schedule})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)
