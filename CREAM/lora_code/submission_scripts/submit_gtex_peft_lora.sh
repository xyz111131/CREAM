config_path=configs/blood_config_model.yaml
config_path_lora=configs/blood_config_lora.yaml
sbatch slurm_train_gtex_peft_lora.sh $config_path $config_path_lora 0 SingleGene
#for model_type in SingleGene MultiGene; do
#    for fold in 0 1 2; do
#        sbatch slurm_train_gtex.sh $config_path $fold $model_type
#    done
#done

