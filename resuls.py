import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()


def get_real(path, subjects):
    real = {"termination": [], "bifurcation": []}

    for subject in subjects:
        inform_path = f"{path}/{subject}/ground_truth_templates/inform.json"

        with open(inform_path, "r") as inform:
            data = json.loads(inform.read())

        real["termination"].append(data['termination'])
        real["bifurcation"].append(data['bifurcation'])

    return real


def get_values(df, subjects, lbl):
    values_count = []
    for f in subjects:
        values = df[df['file'] == f][lbl].value_counts()
        values_count.append(values.to_dict())

    return values_count


def plot_by_kind_percentages(df, subjects, real, output):
    values = get_values(df, subjects, 'kind')
    terminations = [x.get('termination', 0) for x in values]
    bifurcations = [x.get('bifurcation', 0) for x in values]

    ind = np.arange(fingerprints_x_subjects)
    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x=ind, height=[t / r * 100 if r > 0 else 0
                          for t, r in zip(terminations, real.get('termination'))],
           width=width, label='% Terminations')
    ax.bar(x=ind + width, height=[b / r * 100 if r > 0 else 0
                                  for b, r in zip(bifurcations, real.get('bifurcation'))],
           width=width, label='% Bifurcations')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(subjects)
    ax.legend()

    plt.savefig(output)
    plt.show()


def plot_by_kind(df, subjects, real, output):
    values = get_values(df, subjects, 'kind')
    terminations = [x.get('termination', 0) for x in values]
    bifurcations = [x.get('bifurcation', 0) for x in values]

    width = 0.2
    ind = np.arange(fingerprints_x_subjects)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(x=ind, height=terminations, width=width, label='Terminations Predictions')
    ax.bar(x=ind + width, height=real.get('termination'), width=width, label='Terminations Ground Truth')
    ax.bar(x=ind + width * 2, height=bifurcations, width=width, label='Bifurcations Predictions')
    ax.bar(x=ind + width * 3, height=real.get('bifurcation'), width=width, label='Bifurcations Ground Truth')
    ax.set_xticks(ind + 1.5 * width)
    ax.set_xticklabels(subjects)
    ax.legend()

    plt.savefig(output)
    plt.show()


def get_pred_reals(df, subjects, real):
    predictions = df['file'].value_counts().to_dict()
    predictions = [predictions.get(s) for s in subjects]
    reals = [x + y for x, y in zip(
        real.get('termination'),
        real.get('termination')
    )]

    return predictions, reals


def plot_by_match(df, subjects, real, output):
    predictions, reals = get_pred_reals(df, subjects, real)
    ind = np.arange(fingerprints_x_subjects)
    width = 0.3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x=ind, height=predictions, width=width, label='Minutiae Prediction')
    ax.bar(x=ind + width, height=reals, width=width, label='Minutiae Ground Truth')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(subjects)
    ax.legend()

    plt.savefig(output)
    plt.show()


def plot_by_match_percentages(df, subjects, real, output):
    predictions, reals = get_pred_reals(df, subjects, real)
    ind = np.arange(fingerprints_x_subjects)
    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x=ind, height=[p / r if r > 0 else 0 for p, r in zip(predictions, reals)],
           width=width, label='% Minutiae')
    ax.set_xticks(ind)
    ax.set_xticklabels(subjects)
    ax.legend()

    plt.savefig(output)
    plt.show()


def plot_results(dfs, dfs_f, subjects, pipeline_results, out_path):
    for subs in subjects:
        df = dfs[dfs['file'].isin(subs)]
        df_f = dfs_f[dfs_f['file'].isin(subs)]

        base_name = f"{out_path}/{subs[0]}_to_{subs[-1]}_{'{}'}.png"

        plot_by_kind(df, subs, get_real(pipeline_results, subs),
                     base_name.format("terms_and_bifs"))
        plot_by_kind_percentages(df, subs, get_real(pipeline_results, subs),
                                 base_name.format("terms_and_bifs_percentages"))
        plot_by_match(df_f, subs, get_real(pipeline_results, subs),
                      base_name.format("all_matches"))
        plot_by_match_percentages(df_f, subs, get_real(pipeline_results, subs),
                                  base_name.format("all_matches_percentages"))


if __name__ == "__main__":

    plot_path = "./plots"
    pipeline_results = './fingerprints_results_pipeline'
    results_path = f'{pipeline_results}/results_matches.csv'
    results_full_path = f'{pipeline_results}/results_matches.csv'

    result_dataframe = pd.read_csv(results_path, sep=",")
    result_full_dataframe = pd.read_csv(results_full_path, sep=",")

    subjects = 10
    fingerprints_x_subjects = 8
    subjects_list = [[f"{100 + i + 1}_{j + 1}"
                      for j in range(fingerprints_x_subjects)]
                     for i in range(subjects)]

    os.makedirs(plot_path, exist_ok=True)
    plot_results(result_dataframe, result_full_dataframe,
                 subjects_list, pipeline_results, plot_path)
