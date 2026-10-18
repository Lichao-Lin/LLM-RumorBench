"""Legacy four-column Excel entry point, with row-aligned missingness."""
import argparse
import pandas as pd
from rumorbench.evaluation import score


def simple_kappa_test(file_path, ordinal_labels=None):
    df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    if df.shape[1] < 4:
        raise ValueError("Expected at least four columns")
    results = []
    for first, second in ((0, 1), (2, 3)):
        pair = df.iloc[:, [first, second]].apply(
            lambda col: col.astype('string').str.strip().replace('', pd.NA)).dropna()
        if pair.empty:
            raise ValueError("No complete pairs remain")
        labels = ordinal_labels if first == 2 and ordinal_labels else sorted(
            set(pair.iloc[:, 0]) | set(pair.iloc[:, 1]))
        if len(labels) < 2:
            raise ValueError("Kappa is undefined for a single observed category")
        if not (set(pair.iloc[:, 0]) | set(pair.iloc[:, 1])) <= set(labels):
            raise ValueError("Observed labels outside --ordinal-labels")
        results.append(score(pair.iloc[:, 0], pair.iloc[:, 1], labels,
                             ordinal=first == 2 and ordinal_labels is not None))
    return results[0]['kappa'], results[1]['kappa'], results[1]['linear_kappa']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Excel file; columns 1–2 and 3–4 are paired')
    parser.add_argument('--ordinal-labels', nargs='+', help='Full order for columns 3–4')
    args = parser.parse_args()
    kappa12, kappa34, weighted34 = simple_kappa_test(args.input, args.ordinal_labels)
    print({'kappa_12': kappa12, 'kappa_34': kappa34, 'linear_kappa_34': weighted34})
