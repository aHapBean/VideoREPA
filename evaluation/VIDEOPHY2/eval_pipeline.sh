INPUT_CSV='VideoREPA_2B.csv'

OUTPUT_FOLDER="./output_dir/${INPUT_CSV%.csv}"

INPUT_CSV="./csv_file/$INPUT_CSV"

mkdir "$OUTPUT_FOLDER"

echo "output_dir:$OUTPUT_FOLDER"

python inference.py --input_csv $INPUT_CSV  \
--checkpoint /mnt/petrelfs/zhangxiangdong/VideoGeneration/CogVideo/finetune/ckpt/videophy2 \
--output_csv "$OUTPUT_FOLDER/output_sa.csv" --task sa

echo "output_dir:$OUTPUT_FOLDER"  

python inference.py --input_csv $INPUT_CSV  \
--checkpoint /mnt/petrelfs/zhangxiangdong/VideoGeneration/CogVideo/finetune/ckpt/videophy2 \
--output_csv "$OUTPUT_FOLDER/output_pc.csv" --task pc

echo "output_dir:$OUTPUT_FOLDER"

cp calculate_mean.py "$OUTPUT_FOLDER/calculate_mean.py"
cd "$OUTPUT_FOLDER"

echo "output_dir:$OUTPUT_FOLDER"

python calculate_mean.py