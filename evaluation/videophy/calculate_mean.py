import csv
import numpy as np

def calculate_means(input_csv):
    third_column_data = []
    
    with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        # next(csvreader)  # Skip header
        
        for row in csvreader:
            third_column_data.append(float(row[2]))
    
    thresholded_data = [0 if x < 0.5 else 1 for x in third_column_data]
    thresholded_mean = np.mean(thresholded_data)
    
    return thresholded_mean, third_column_data  

sa_thresholded, sa_data = calculate_means('./videocon_physics_sa_testing.csv')
pc_thresholded, pc_data = calculate_means('./videocon_physics_pc_testing.csv')

print(f'sa thresholded mean: {sa_thresholded}')
print(f'pc thresholded mean: {pc_thresholded}')

assert len(sa_data) == len(pc_data), "NOTE the length of sa and pc score should be the same"

# Save results
with open(f'sa_{sa_thresholded:.4f}_pc_{pc_thresholded:.4f}.txt', "w") as f:
    f.write(f'sa thresholded mean: {sa_thresholded}\n')
    f.write(f'pc thresholded mean: {pc_thresholded}\n')
    f.write(f'{sa_thresholded:.5f} {pc_thresholded:.5f}\n')


results = {
        'solid_solid': {},
        'solid_fluid': {},
        'fluid_fluid': {}
}

with open("videophy.txt", "r", encoding="utf-8") as f:
    captions = [line.strip() for line in f if line.strip()]
import pandas
def calculate_means_fluid(input_csv):
    third_column_data = {
        'solid_solid': [],
        'solid_fluid': [],
        'fluid_fluid': []
    }
    df = pandas.read_csv("reference.csv")
    with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        # next(csvreader)  # Skip header
        
        for idx, row in enumerate(csvreader):
            caption = captions[idx]
            
            filtered_df = df[df['caption'] == caption]
            label = filtered_df.iloc[0]['states_of_matter']
            assert label is not None 
            third_column_data[label].append(float(row[2]))
    for label in third_column_data.keys():
        data_data = third_column_data[label]
        data_data = [0 if x < 0.5 else 1 for x in data_data]
        if 'pc' in input_csv:
            results[label]['pc'] = np.mean(data_data)
        else:
            results[label]['sa'] = np.mean(data_data)


calculate_means_fluid('./videocon_physics_sa_testing.csv')
calculate_means_fluid('./videocon_physics_pc_testing.csv')

with open(f'sa_{sa_thresholded:.4f}_pc_{pc_thresholded:.4f}.txt', "a") as f:
    for key in results.keys():
        f.write(f"{key}: {results[key]['sa']:.5f} {results[key]['pc']:.5f}\n")
        print(f"{key}: {results[key]['sa']:.5f} {results[key]['pc']:.5f}")
