import os

def save_config(args):
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    args_dict = vars(args)
    config_file = args.output + '/config.csv'
    with open(config_file, 'w') as f:
        for key in args_dict.keys():
            f.write('{}, {}\n'.format(key, args_dict[key]))