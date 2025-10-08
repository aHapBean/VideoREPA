import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from finetune.models.utils import get_model_cls
from finetune.schemas import Args


def main():
    args = Args.parse_args()
    
    if args.comment != "":
        args.comment = "_" + args.comment
    
    if 'openvid' in args.train_data_path[0]:
        dataset_name = 'OpenVid'
    else:
        raise ValueError("please note the train json should contain dataset name (openvid)")
    print(args.precomputing)
    
    if args.align_models[0] != 'VideoMAEv2':
        args.output_dir = args.output_dir.parent / (args.output_dir.stem + '_' + args.model_name + '_' + args.loss + args.comment + f'_{args.align_models[0]}')
    else:    
        args.output_dir = args.output_dir.parent / (args.output_dir.stem + '_' + args.model_name + '_' + args.loss + args.comment)

    trainer_cls = get_model_cls(args.model_name, args.training_type)
    
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
