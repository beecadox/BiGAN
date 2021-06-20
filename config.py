def get_config(phase):
    config = Config(phase)
    return config


class Config(object):

    def __init__(self, phase):


        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
