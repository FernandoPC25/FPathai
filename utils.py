import os
import csv

def generate_csv_from_h5_files(folder_path, output_csv_path, label):
    with open(output_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Path', 'WSI_name', 'Patient_ID', 'Label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(folder_path):
            if filename.endswith('.svs'):
                full_path = os.path.join(folder_path, filename)
                patient_id = full_path.split('/')[-1].split('-')[:3]
                patient_id = '-'.join(patient_id)
                file_name = os.path.splitext(filename)[0]
                file_name = file_name + '.h5'

                writer.writerow({'Path': full_path, 'WSI_name': file_name, 'Patient_ID': patient_id, 'Label': label})
                
# Usage of the function
folder_path = '/home/fernandopc/Documentos/prueba10/control'
output_csv_path2 = '/home/fernandopc/Documentos/prueba10/control/control.csv' 

generate_csv_from_h5_files(folder_path, output_csv_path2, label="Control")

# Usage of the function
folder_path = '/home/fernandopc/Documentos/prueba10/tumor/'
output_csv_path2 = '/home/fernandopc/Documentos/prueba10/control/tumor.csv' 

generate_csv_from_h5_files(folder_path, output_csv_path2, label="Tumor")



import os
import shutil

# para mover todas las imagenes svs a un mismo sitio (que no estén cada una en una carpeta)
#carpeta_principal = '/home/fernandopc/Documentos/prueba1'
carpeta_principal = '/home/fernandopc/Escritorio/WSI-Pancreas-Classification/imagen/imagenessvs2/carpeta_patches_control'

# Ruta de la carpeta donde se guardarán las imágenes .svs
carpeta_destino = '/home/fernandopc/Escritorio/WSI-Pancreas-Classification/imagen/imagenessvs2/carpeta_patches_control'

# Crear la carpeta de destino si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Recorrer las carpetas
for carpeta in os.listdir(carpeta_principal):
    ruta_carpeta = os.path.join(carpeta_principal, carpeta)

    # Comprobar si es una carpeta
    if os.path.isdir(ruta_carpeta):
        # Recorrer los archivos de la carpeta
        for archivo in os.listdir(ruta_carpeta):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)

            # Comprobar si es un archivo .svs
            if os.path.isfile(ruta_archivo) and archivo.endswith('.h5'):
                # Copiar el archivo a la carpeta de destino
                shutil.copy2(ruta_archivo, carpeta_destino)
                
                
                
                
                
                
                
# pasar de carpetas a una simple carpeta                
import os
import shutil

# Path to folder of patches
patch_path = '/home/fernandopc/Documentos/prueba4/pruebasa'

# Move images to folder "A" and delete empty folders
for folder in os.listdir(patch_path):
    folder_path = os.path.join(patch_path, folder)
    h5_files = [file for file in os.listdir(folder_path) if file.endswith(".h5")]
    
    for h5_file in h5_files:
        source_path = os.path.join(folder_path, h5_file)
        destination_path = os.path.join(patch_path, h5_file)
        shutil.move(source_path, destination_path)
    
    os.rmdir(folder_path)

print("Process completed.")               
                
                
                
                
                
                
                
