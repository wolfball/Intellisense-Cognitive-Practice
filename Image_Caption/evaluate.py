import argparse
import pandas as pd

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from utils.util import ptb_tokenize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file", type=str)
    parser.add_argument("reference_file", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()
    print(args.prediction_file)
    prediction_df = pd.read_json(args.prediction_file)
    #[n, var_len]
    key_to_pred = dict(zip(prediction_df["img_id"], prediction_df["prediction"]))
    #[n, 5, var_len]
    captions = open(args.reference_file, 'r').read().strip().split('\n')
    key_to_refs = {}
    for i, row in enumerate(captions):
        row = row.split('\t')
        row[0] = row[0][:len(row[0]) - 2]  # filename#0 caption
        if row[0] not in key_to_pred:
            continue
        if row[0] in key_to_refs:
            key_to_refs[row[0]].append(row[1])
        else:
            key_to_refs[row[0]] = [row[1]]

    scorers = [Bleu(n=4), Rouge(), Meteor(), Cider(), Spice()]
    key_to_refs = ptb_tokenize(key_to_refs)
    key_to_pred = ptb_tokenize(key_to_pred)
    id_map = list(key_to_refs.keys())
    spider_scores = list()
    output = {"SPIDEr": 0}
    with open(args.output_file, "w") as writer:
        for scorer in scorers:
            score, scores = scorer.compute_score(key_to_refs, key_to_pred)
            method = scorer.method()
            output[method] = score
            if method == "Bleu":
                for n in range(4):
                    print(f"===== Bleu-{n+1} =====", file=writer)
                    print("Bleu-{}: {:.3f}".format(n + 1, score[n]), file=writer)
                    best_score, worst_score = max(scores[n]), min(scores[n])
                    print(f"----- best-{best_score:.3f} -----", file=writer)
                    imgid = id_map[scores[n].index(best_score)]
                    print(f"img:{imgid}", file=writer)
                    print(key_to_refs[imgid], file=writer)
                    print(key_to_pred[imgid], file=writer)
                    print(f"----- worst-{worst_score:.3f} -----", file=writer)
                    imgid = id_map[scores[n].index(worst_score)]
                    print(f"img:{imgid}", file=writer)
                    print(key_to_refs[imgid], file=writer)
                    print(key_to_pred[imgid], file=writer)
            elif method == 'SPICE':
                scores = [s['All']['f'] for s in scores]
                print(f"===== {method} =====", file=writer)
                print(f"{method}: {score:.3f}", file=writer)
                best_score, worst_score = max(scores), min(scores)
                print(f"----- best-{best_score:.3f} -----", file=writer)
                imgid = id_map[scores.index(best_score)]
                print(f"img:{imgid}", file=writer)
                print(key_to_refs[imgid], file=writer)
                print(key_to_pred[imgid], file=writer)
                print(f"----- worst-{worst_score:.3f} -----", file=writer)
                imgid = id_map[scores.index(worst_score)]
                print(f"img:{imgid}", file=writer)
                print(key_to_refs[imgid], file=writer)
                print(key_to_pred[imgid], file=writer)
            else:
                scores = list(scores)
                print(f"===== {method} =====", file=writer)
                print(f"{method}: {score:.3f}", file=writer)
                best_score, worst_score = max(scores), min(scores)
                print(f"----- best-{best_score:.3f} -----", file=writer)
                imgid = id_map[scores.index(best_score)]
                print(f"img:{imgid}", file=writer)
                print(key_to_refs[imgid], file=writer)
                print(key_to_pred[imgid], file=writer)
                print(f"----- worst-{worst_score:.3f} -----", file=writer)
                imgid = id_map[scores.index(worst_score)]
                print(f"img:{imgid}", file=writer)
                print(key_to_refs[imgid], file=writer)
                print(key_to_pred[imgid], file=writer)

            if method in ["CIDEr", "SPICE"]:
                spider_scores.append(scores)
                output["SPIDEr"] += score
        output["SPIDEr"] /= 2
        spider_scores = [(spider_scores[0][i]+spider_scores[1][i])/2 for i in range(len(spider_scores[0]))]
        print(f"===== SPIDEr =====", file=writer)
        print(f"SPIDEr: {output['SPIDEr']:.3f}", file=writer)
        best_score, worst_score = max(spider_scores), min(spider_scores)
        print(f"----- best-{best_score:.3f} -----", file=writer)
        imgid = id_map[spider_scores.index(best_score)]
        print(f"img:{imgid}", file=writer)
        print(key_to_refs[imgid], file=writer)
        print(key_to_pred[imgid], file=writer)
        print(f"----- worst-{worst_score:.3f} -----", file=writer)
        imgid = id_map[spider_scores.index(worst_score)]
        print(f"img:{imgid}", file=writer)
        print(key_to_refs[imgid], file=writer)
        print(key_to_pred[imgid], file=writer)