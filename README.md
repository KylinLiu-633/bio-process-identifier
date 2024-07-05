# Mention-agnostic biological process identification

This program is for process identification according to the HOIP ontology on the HOIP dataset.

## Binary classifier based on BERT

Put your data in the "../data/for_binary_classification" folder, and the naming and format should be as the examples in
the folder after you download HOIP dataset.

We have tried 3 pretrained language model,

BioBERT: "dmis-lab/biobert-v1.1"

PubMedBERT: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

SciBERT: "allenai/scibert_scivocab_uncased"

Default hyperparameters:

```json
{
  "transformer_name": "dmis-lab/biobert-v1.1",
  "data_dir": "../data/for_binary_classification",
  "model_name": "biobert-8",
  "learning_rate": 15e-6,
  "num_train_epochs": 10,
  "per_device_train_batch_size": 8,
  "per_device_eval_batch_size": 8,
  "max_length": 512,
  "do_negative_sampling": true,
  "neg_ratio": 8,
  "training": true,
  "evaluation": true,
  "prediction": false
}
```

Enter in the "scripts" folder first.

Please set the model card of pretrained language model and model name, then run the following command for training and
evaluation.

Details of hyperparameters can be found in line 430-445 in "scripts/binary_classification.py".

```commandline
python binary_classification.py -data_dir for_binary_classification -transformer_name dmis-lab/biobert-v1.1 -neg_ratio 8
```

After the training/evaluation/prediction finished, the model is stored in "scripts/model/model-name" and the model output would be
in "scripts/logs/model-name".

The model output is organized in a json file, while each line is in a format like:
```json
{"doc_key": "ID72", "ent_key": "http://purl.obolibrary.org/obo/GO_0070269", "predict": 1, "truth": 1}
```

### Get predicted entity given model output

The model outputs are stored in "scripts/logs/model-name".

Please complete the file "../data/meta/entityid_to_label.json" after downloading the HOIP dataset.

Then enter in "../scripts/utils".

Run the command, will output the predictions of every file in the log_dir to the output_dir.

```commandline
python post_for_bi_cls.py -log_dir ../logs/model-name -output_dir ../../data/after_binary_classification/model-name
```

