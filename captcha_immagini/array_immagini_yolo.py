# Script che crea un file json contenente i percorsi 
# delle immagini da utilizzare nella pagina web della demo

import os
import json

# costanti
base_directory = 'captcha_immagini'  # cartella madre
directory_path = os.path.join(base_directory, 'immagini/images')  # cartella che contiene le immagini
allowed_subdirs = [ # Elenco delle cartelle da considerare
    'Bycicle',
    'Bus',
    'Car',
    'Hydrant',
    'Traffic Light',
    'Motorcycle'
] 

def get_all_paths(directory, base_directory, allowed_subdirs):
    paths = []
    # os.walk scorre la directory e le sue sottodirectory
    for root, _, files in os.walk(directory):
        # Verifica se il percorso corrente è una delle cartelle da considerare
        relative_root = os.path.relpath(root, base_directory)
        if any(relative_root.startswith(os.path.join('immagini/images', subdir)) for subdir in allowed_subdirs):
            for file in files:
                # Costruisce il percorso completo per ciascun file
                full_path = os.path.join(root, file)
                # Calcola il percorso relativo alla cartella base
                relative_path = os.path.relpath(full_path, base_directory)
                paths.append(relative_path)
    return paths


all_paths = get_all_paths(directory_path, base_directory, allowed_subdirs)
print(all_paths)

# Scrivi l'array su un file JSON
with open('files.json', 'w') as f:
    json.dump(all_paths, f, indent=4) 