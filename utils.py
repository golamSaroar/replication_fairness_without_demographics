from argparse import Namespace
import main
import numpy as np

performance_metrics_list = ['micro_avg_auc', 'macro_avg_auc', 'min_auc', 'minority_auc', 'accuracy']


def get_results(seed, conf, optimal_hyper_parameters, experiments):
    result_per_seed = {}

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

        result_per_seed[(dataset, model)] = main.train_and_evaluate(current_conf)

    return result_per_seed


def convert_result_to_dict(results, experiments, performance_metrics_list):
    return {
        k: {
            metric: {
                'mean': np.mean([result_dict[k][metric] for result_dict in results]),
                'std': np.std([result_dict[k][metric] for result_dict in results])
            } for metric in performance_metrics_list
        } for k in experiments
    }
