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
