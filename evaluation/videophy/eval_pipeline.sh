INPUT_CSV='VideoREPA_2B.csv'


OUTPUT_FOLDER="./output_dir/${INPUT_CSV%.csv}"

mkdir -p "$OUTPUT_FOLDER"

echo "output_dir: $OUTPUT_FOLDER"

python utils/prepare_data.py --input_csv "./csv_file/$INPUT_CSV" --output_folder "$OUTPUT_FOLDER"

python videocon/training/pipeline_video/entailment_inference.py \
--input_csv "$OUTPUT_FOLDER/sa_testing.csv" --output_csv "$OUTPUT_FOLDER/videocon_physics_sa_testing.csv" --checkpoint /mnt/petrelfs/zhangxiangdong/VideoGeneration/benchmark/videophy/model

python videocon/training/pipeline_video/entailment_inference.py \
--input_csv "$OUTPUT_FOLDER/physics_testing.csv" --output_csv "$OUTPUT_FOLDER/videocon_physics_pc_testing.csv" --checkpoint /mnt/petrelfs/zhangxiangdong/VideoGeneration/benchmark/videophy/model

cp calculate_mean.py "$OUTPUT_FOLDER/calculate_mean.py"
cp videophy.txt "$OUTPUT_FOLDER/videophy.txt"
cp reference.csv "$OUTPUT_FOLDER/reference.csv"
cd $OUTPUT_FOLDER
python calculate_mean.py