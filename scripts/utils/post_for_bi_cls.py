import argparse
import json
import os


def get_bi_cls_res(file_dir: str, output_dir: str):
    with open("../../data/meta/entityid_to_label.json", "r") as f_in:
        id2label = json.load(f_in)
        f_in.close()

    for f in os.listdir(file_dir):
        with open(file_dir + "/" + f, "r") as f_in:
            inst_res = []
            for line in f_in.readlines():
                inst_res.append(json.loads(line))
            f_in.close()

        res_per_des_out = {}
        for d in inst_res:
            if d["doc_key"] not in res_per_des_out.keys():
                res_per_des_out[d["doc_key"]] = {}
            if d["predict"] == 1:
                if d["ent_key"] not in res_per_des_out[d["doc_key"]].keys():
                    res_per_des_out[d["doc_key"]][d["ent_key"]] = ""

        final_out = {}
        if f.startswith("eval-"):
            with open("../../data/for_binary_classification/test.json", "r") as f_in:
                gold_data = json.load(f_in)
                f_in.close()
            for d_id, d in gold_data.items():
                final_out[d_id] = {"Description": d["Description"], "Gold": d["Gold_Entity"], "Prediction": []}
        else:
            with open("../../data/for_binary_classification/pred.json", "r") as f_in:
                gold_data = json.load(f_in)
                f_in.close()
            for d_id, d in gold_data.items():
                final_out[d_id] = {"Description": d["Description"], "Gold": [], "Prediction": []}

        for d_id, d in res_per_des_out.items():
            for e_id, e in d.items():
                final_out[d_id]["Prediction"].append({"id": e_id, "name": id2label[e_id]})
        with open(f"{output_dir}/{f}.json", "w") as f_out:
            json.dump(final_out, f_out)
            f_out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-log_dir", type=str, default="../logs/biobert-8", help="path of your model output")
    parser.add_argument("-output_dir", type=str, default="../../data/after_binary_classification/biobert-8",
                        help="path of results after post-processing")

    args, _1 = parser.parse_known_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    get_bi_cls_res(file_dir=args.log_dir, output_dir=args.output_dir)
    print("finished.")
