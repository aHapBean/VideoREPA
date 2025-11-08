import pandas as pd

def calculate_mean_and_ratio(input_csv, column_index=3):
    df = pd.read_csv(input_csv)
    mean_value = df.iloc[:, column_index].mean()
    # ratio_2 = (df.iloc[:, column_index] >= 2.0).mean()
    ratio_3 = (df.iloc[:, column_index] >= 3.0).mean()
    ratio_4 = (df.iloc[:, column_index] >= 4.0).mean()
    ratio_5 = (df.iloc[:, column_index] >= 5.0).mean()
    # ratio = (df.iloc[:, column_index] >= 4.0).mean()
    return mean_value, None, ratio_3, ratio_4, ratio_5

def calculate_both_gte_4(input_csv_sa, input_csv_pc):
    df_sa = pd.read_csv(input_csv_sa)
    df_pc = pd.read_csv(input_csv_pc)
    
    both_gte_4 = (df_sa.iloc[:, 3] >= 4.0) & (df_pc.iloc[:, 3] >= 4.0)
    
    matching_rows = df_sa[both_gte_4]['caption']
    proportion = both_gte_4.mean()
    
    return proportion, matching_rows

input_csv_sa = 'output_sa.csv'
mean_value1, ratio2, ratio3, ratio4, ratio5 = calculate_mean_and_ratio(input_csv_sa)
print(f'sa {mean_value1:.3f}, {ratio3 * 100:.2f}% {ratio4 * 100:.2f}% {ratio5 * 100:.2f}%')

input_csv_pc = 'output_pc.csv'
mean_value2, ratio22, ratio23, ratio24, ratio25 = calculate_mean_and_ratio(input_csv_pc)
print(f'pc {mean_value2:.3f}, {ratio23 * 100:.2f}% {ratio24 * 100:.2f}% {ratio25 * 100:.2f}%')

