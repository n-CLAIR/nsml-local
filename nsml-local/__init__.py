# values from nsml
SESSION_ID = ''
SESSION_NAME = ''
GPU_NUM = 0

IS_DATASET = False
HAS_DATASET = False
DATASET_PATH = ''
DATASET_NAME = ''
IS_ON_NSML = False


def report(*args, **kwargs):
	pass


def paused(*args, **kwargs):
	pass


def save(*args, **kwargs):
	pass


def bind(save=None, load=None, infer=None, *args, **kwargs):
    global user_infer
    if infer:
        user_infer = infer
    elif 'infer' in kwargs:
        user_infer = kwargs['infer']


def infer(data, *args, **kwargs):
    return user_infer(data)


def cache(*args, **kwargs):
	pass


class Visdom:
	pass
