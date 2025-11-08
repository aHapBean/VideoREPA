import csv
target_ver = "VideoREPA_2B"
prefix = f'/Path/to/Your/Videos/{target_ver}/videophy/'


with open('all.txt', 'r', encoding='utf-8') as file_a, open(f'./csv_file/{target_ver}.csv', 'w', encoding='utf-8') as file_b:
    csv_writer = csv.writer(file_b)
    csv_writer.writerow(['videopath', 'caption'])
    
    for line in file_a:
        words = line.strip().strip('.').split(' ')
        new_line = prefix + '_'.join(words) + '.mp4'
        caption = line.strip().split('.')[0]  
        csv_writer.writerow([new_line, caption])

