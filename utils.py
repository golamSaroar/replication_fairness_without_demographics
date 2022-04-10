from argparse import Namespace
import main


def get_results(seed, conf, optimal_hyper_parameters, experiments):
    result_dict = {}

    for dataset, model in experiments:
        current_conf = Namespace(**vars(conf))
        if model == 'IPW(S)':
            current_conf.model = 'IPW'
            current_conf.sensitive_label = False
        elif model == 'IPW(S+Y)':
            current_conf.model = 'IPW'
            current_conf.sensitive_label = True
        else:
            current_conf.model = model
        current_conf.dataset = dataset
        current_conf.seed = seed

        if optimal_hyper_parameters is not None:
            for k, v in optimal_hyper_parameters[dataset][model].items():
                setattr(current_conf, k, v)

        # train and evaluate the model
        result_dict[(dataset, model)] = main.train_and_evaluate(current_conf)

    return result_dict
