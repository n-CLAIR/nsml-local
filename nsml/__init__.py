# values from nsml

# number of gpus
GPU_NUM = 0
#
DATASET_PATH = ''
DATASET_NAME = ''
# HAS_DATASET is True if dataset_path is not None
HAS_DATASET = False
# IS_ON_NSML is True if model is running on nsml server.
IS_ON_NSML = False

"""
below functions describe how to communicate nsml server. 
So, only working correctly on nsml server
"""


def report(summary=False, scope=None, **kwargs):
    """
    Track changes in variable values, such as loss and optimizers and so on.
    also use them for tensorboard, visdom (using, nsml plot [session])
    If you pass a value that can not be serialized by json, an error occurs. (such as TensorType…)

    :param summary: If True, the values of these variables are shown in 'nsml ps'.

    :param scope: If scope=locals(),
                    You can access variables in that scope using the 'nsml exec [session] [variable]'

    :param kwargs: You can set the variables that you shown in tensorboard or visdom.

    :example:
        nsml.report(
                    summary=False,
                    epoch=epoch+int(config.iteration),
                    epoch_total=config.epochs,
                    iter=iter_idx,
                    iter_total=total_length,
                    batch_size=batch_size,
                    train__loss=running_loss / num_runs,
                    train__accuracy=running_acc / num_runs,
                    step=epoch*total_length+iter_idx,
                    lr = opt_params['lr'],
                    scope=locals()
                    )
    """
    pass


def paused(scope=None):
    """
    Switch to a wait state that waits until nsml command is passed.

    In case of fork command executed,
        The load function is executed and the next train is executed.

    In case of infer command executed,
        The load, infer function are called in order.
        After That, the state is changed to a waiting state until the input is received.
        (Infer sessions remain forever, so please delete them manually)

    In case of submit command executed,
        The load, infer function are called in order.
        After evaluate test_data using infer function, The result is submitted to the server.
        Session is deleted after submitted.

    :param scope:
        It should be main function’s scope.
        Otherwise, The scope may not be properly captured and the load function may not be called properly.
        (See example7, 9,10,11)

    :example:
        if config.pause:
            nsml.paused(scope=scope)

    """
    pass


def save(iteration=None, save_fn=None):
    """
    Save the model or optimizer on the storage server.
    passing the object to bind_model() function, It can access that variables and save as a file.

    :param iteration: recognize model’s identity. If None, It is set to the current time.

    :param save_fn: The save function. If None, The function that is bound in nsml.bind() is called.
                    If nothing is defined, the default save function is called.

    :example:
        def save(filename, **kwargs):
            torch.save(object, filename)
            print('saved')

        nsml.save(iteration=iteration, save_fn=save)

    """
    pass


def bind(save=None, load=None, infer=None, **kwargs):
    """
    Defines the save, load and infer functions used by nsml.
    The default functions are set, if save, load arguments are None.(infer has no default function.)
    With default functions, nsml.bind() can receive which objects are saved or load in ‘kwargs’ arguments.


    :param save: Save Objects. When the nsml.save() function is called,
                 The save function defined in bind_model is called.
                 and saved objects are transferred to nsml internal storage server.

    :param load: Defines how to load the saved objects. It is used for nsml fork,infer,submit.

    :param infer: When nsml infer, submit command is called, infer function called.
    :param kwargs: Used to pass arguments to save or load function. see below

    :example:
        def load(filename):

            torch.load(filename)
            ...

        def save(filename, **kwargs):
            if kwargs[‘real’]:
                torch.save(object,filename)

        def infer(data, **kwargs):
            # Defines How to infer using saved objects. It works differently depending on the command (submit, infer)
            model.eval()
            output_predictions = model(data)
            return list(zip(prob, prediction.tolist()))
            ...


        # kwargs got a 'real=True' variable.
        nsml.bind(save=save, load=load, infer=infer, real=True)

    """
    global user_infer
    if infer:
        user_infer = infer
    elif 'infer' in kwargs:
        user_infer = kwargs['infer']


def infer(data, **kwargs):
    """
    Defines How to infer using saved objects. It works differently depending on the command(submit, infer).
    This function should be implemented and bound in 'nsml.bind(infer=infer)'

    :param data:
        The input data.
             In case of submit:
                 input data from ‘data_loader’ which is defined by dataset owner.
             In case of infer:
                 Depending on the app_type, It receives different data format.
                     app_type(tobe added):
                          default(None):   {‘data’:[input_data], ‘label’: None}
                          list_string:         {‘data’: [[input_data]])

    :param kwargs:
        It is arguments that pass by "nsml exec [session] 'nsml.infer(data, app_type="list_string", kwargs=…)' "
    :return:
        In case of submit:
            output value should be match with below format for using Leaderboard.
                [probability, prediction]
            evaluation metric only use 'prediction' value, but format should be matched
            see example 10,

        In case of infer:
             The return value you want to inferring is displayed as is.
    """

    return user_infer(data)


def cache(preprocess_fn, **kwargs):
    """
    Caching preprocessing data. It caches the output of preprocess_fn.

    :param preprocess_fn:
        It should be a function that takes a 'output_path' file_name list argument.
        By default, ['./processed'] is set to output_path, If 'output_path' argument not specified.
    :param kwargs:
        arguments pass to ‘preprocess_fn’.

    :example:

        def preprocess(output_path, data):
            data_set = {
                'train': _normalize_image(data['train']['data'], data['train']['label'], transform),
                'test': _normalize_image(data['test']['data'], data['test']['label'], transform)
            }
            with open(output_path[0], 'wb') as file:
                torch.save(data_set, file)

        nsml.cache(preprocess, output_path=['./preprocess.pt'], data=data_loader(DATASET_PATH))

        Then, output_path files([./preprocess.pt]) will be cached.  for more specific details, see example 10

    """

    pass


def copy(source=None, target=None):
    """
    Used to load call by value objects that saved by pickle.
    (Dictionary or Class attributes are copied. list type is not supported)
    make sure source and target types are same before invoking this function.

    :param source: source object
    :param target: target object

    :example:

        class ClasToSave:
            def __init___(self):
                self.elem = 0
                self.elem2 = 1

        class_to_save=ClassToSave()


        def bind_model(model, class_to_save):

            def save(filename, **kwargs):
                with open(os.path.join(filename, 'class.pkl'), 'wb') as fp:
                    pickle.dump(class_to_save, fp)

            def load(filename, **kwargs):
                with open(os.path.join(filename, 'class.pkl, 'rb') as fp:
                    temp_class = pickle.load(fp)
                assert type(temp_class) is type(class_to_save)
                nsml.copy(temp_class, class_to_save)
            ...

            nsml.bind(save=save, load=load)

        bind_model(model=model, class_to_save=class_to_save)


    for more example details, see example 10
    """
    pass


def load(iteration, load_fn=None, session=None):
    """Used to load the model by session name.

    :param iteration: The checkpoin of the model you want to load
    :param load_fn:  A defined function that loads a saved model. If None, It will call bounded load function (nsml.bind())
    :param session:  session name you want to load. If None, It is set to the current session name.

    :example:
        nsml.load(iteration='0', session='KR18712/mnist/1701')
    """


def Visdom(visdom, **kwargs):
    """
    Same as default Visdom class except '_send' method which is sending data to nsml visdom server

    :param visdom:
    :param kwargs:
    :example:

        import visdom
        vis = nsml.Visdom(visdom=vsdom)

    """
    class MyVisdom(visdom.Visdom):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    return MyVisdom(**kwargs)
