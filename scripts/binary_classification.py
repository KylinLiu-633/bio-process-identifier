# Libraries
import argparse
import os
import evaluate
import random
import torch
import json
import torch.optim as optim

from tqdm import tqdm
from transformers import BertTokenizerFast, BertConfig
from torch.utils.data import DataLoader
from networks.bi_cls import BertBinaryClassifier
from datasets import Dataset, DatasetDict

from sklearn.metrics import precision_recall_fscore_support, classification_report

# Model parameter
MAX_SEQ_LEN = 512
PAD_INDEX = 0
UNK_INDEX = 101


def load_bi_cls_data(source_dir: str, tokenizer, max_seq_length=512,
                     do_negative_sampling=True, neg_ratio=8, test_only=False):
    final_data = {}
    max_src_len = -1
    cate_list = ["train", "dev", "test"] if not test_only else ["test"]

    all_ent = []
    for cate in ["train", "dev", "test"]:

        with open(source_dir + "/" + cate + ".json", "r") as f_in:
            data = json.load(f_in)
            f_in.close()
        for d_id, d in data.items():
            for e in d["Gold_Entity"]:
                all_ent.append((e["id"], e["name"]))
    all_ent = list(set(all_ent))

    for cate in cate_list:
        showed_pair = []
        all_insts = []
        doc_count = 0

        with open(source_dir + "/" + cate + ".json", "r") as f_in:
            data = json.load(f_in)
            f_in.close()

        for d_id, d in data.items():

            sent = d["Description"].strip().split(" ")
            new_sent = []
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if len(tokens_wordpiece) == 0:
                    tokens_wordpiece = [tokenizer.unk_token]
                new_sent.extend(tokens_wordpiece)

            input_ids = tokenizer.convert_tokens_to_ids(new_sent)

            for e in d["Gold_Entity"]:
                new_ent = []
                ent_label = e["name"].strip().split(" ")
                for i_t, token in enumerate(ent_label):
                    tokens_wordpiece = tokenizer.tokenize(token)
                    if len(tokens_wordpiece) == 0:
                        tokens_wordpiece = [tokenizer.unk_token]
                    new_ent.extend(tokens_wordpiece)

                ent_input_ids = tokenizer.convert_tokens_to_ids(new_ent)
                new_ids = input_ids + [tokenizer.sep_token_id] + ent_input_ids
                new_ids = new_ids[:max_seq_length - 2]
                ent_input_ids = tokenizer.build_inputs_with_special_tokens(new_ids)

                new_inst = {"input_ids": ent_input_ids,
                            "doc_key": d_id,
                            "ent_key": e["id"],
                            "label": 1}

                all_insts.append(new_inst)
                showed_pair.append((d_id, e["id"]))

                del new_inst

            candidate_neg_ent = []
            for e_id, e_name in all_ent:
                if (d_id, e_id) not in showed_pair:
                    candidate_neg_ent.append((e_id, e_name))

            if do_negative_sampling and cate == "train":
                positive_inst_num = len(d["Gold_Entity"])
                random.shuffle(candidate_neg_ent)
                candidate_neg_ent = candidate_neg_ent[:int(positive_inst_num * neg_ratio)]

            # print("neg:", len(candidate_neg_ent))

            for e_id, e_name in candidate_neg_ent:
                new_ent = []
                ent_label = e_name.strip().split(" ")
                # ent_input_ids = ""
                for i_t, token in enumerate(ent_label):
                    tokens_wordpiece = tokenizer.tokenize(token)
                    if len(tokens_wordpiece) == 0:
                        tokens_wordpiece = [tokenizer.unk_token]
                    new_ent.extend(tokens_wordpiece)

                ent_input_ids = tokenizer.convert_tokens_to_ids(new_ent)
                new_ids = input_ids + [tokenizer.sep_token_id] + ent_input_ids
                new_ids = new_ids[:max_seq_length - 2]
                ent_input_ids = tokenizer.build_inputs_with_special_tokens(new_ids)
                # print(ent_input_ids)

                new_inst = {"input_ids": ent_input_ids,
                            "doc_key": d_id,
                            "ent_key": e_id,
                            "label": 0}

                all_insts.append(new_inst)
                showed_pair.append((d_id, e_id))
                del new_inst

        final_data[cate] = Dataset.from_list(all_insts)

        src_lens = [len(c["input_ids"]) for c in all_insts]
        max_src_len = max(max_src_len, max(src_lens))

    for k, v in final_data.items():
        print(f"Load in {len(v)} instances for {k} set, containing {sum(c['label'] for c in v)} positive instances.")

    return DatasetDict(final_data), max_src_len


def load_bi_cls_data_pred(source_dir: str, tokenizer, max_seq_length=512):
    final_data = {}
    max_src_len = -1
    cate_list = ["pred"]
    all_ent = []

    for cate in ["train", "dev", "test"]:

        with open(source_dir + "/" + cate + ".json", "r") as f_in:
            data = json.load(f_in)
            f_in.close()
        for d_id, d in data.items():
            for e in d["Gold_Entity"]:
                all_ent.append((e["id"], e["name"]))
    all_ent = list(set(all_ent))

    for cate in cate_list:
        showed_pair = []
        all_insts = []

        with open(source_dir + "/" + cate + ".json", "r") as f_in:
            data = json.load(f_in)
            f_in.close()

        for d_id, d in data.items():

            sent = d["Description"].strip().split(" ")
            new_sent = []
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if len(tokens_wordpiece) == 0:
                    tokens_wordpiece = [tokenizer.unk_token]
                new_sent.extend(tokens_wordpiece)

            input_ids = tokenizer.convert_tokens_to_ids(new_sent)

            candidate_neg_ent = []
            for e_id, e_name in all_ent:
                if (d_id, e_id) not in showed_pair:
                    candidate_neg_ent.append((e_id, e_name))

            for e_id, e_name in candidate_neg_ent:
                new_ent = []
                ent_label = e_name.strip().split(" ")
                for i_t, token in enumerate(ent_label):
                    tokens_wordpiece = tokenizer.tokenize(token)
                    if len(tokens_wordpiece) == 0:
                        tokens_wordpiece = [tokenizer.unk_token]
                    new_ent.extend(tokens_wordpiece)

                ent_input_ids = tokenizer.convert_tokens_to_ids(new_ent)
                new_ids = input_ids + [tokenizer.sep_token_id] + ent_input_ids
                new_ids = new_ids[:max_seq_length - 2]
                ent_input_ids = tokenizer.build_inputs_with_special_tokens(new_ids)

                new_inst = {"input_ids": ent_input_ids,
                            "doc_key": d_id,
                            "ent_key": e_id}

                all_insts.append(new_inst)
                showed_pair.append((d_id, e_id))
                del new_inst

        final_data[cate] = Dataset.from_list(all_insts)

        src_lens = [len(c["input_ids"]) for c in all_insts]
        max_src_len = max(max_src_len, max(src_lens))

    for k, v in final_data.items():
        print(f"Load in {len(v)} instances for {k} set, containing {sum(c['label'] for c in v)} positive instances.")

    return DatasetDict(final_data), max_src_len


# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):
    if save_path is None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path is None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return accuracy.compute(predictions=predictions, references=labels)


# Training Function

def do_train(model, optimizer, train_loader, dev_loader, eval_every, num_epochs, model_file_path,
             model_name, device, best_valid_accu=-1):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    pbar = tqdm(total=len(train_loader) * num_epochs)
    # training loop
    model.train()
    for epoch in range(num_epochs):

        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, doc_ids, ent_ids, labels = batch
            inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device),
                'labels': labels.to(device),
            }
            output = model(**inputs)
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1
            pbar.update(1)

            del input_ids, attention_mask, doc_ids, ent_ids, labels, batch, output

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                truths = []
                predictions = []

                with torch.no_grad():

                    # validation loop
                    for v_input_ids, v_attention_mask, doc_ids, ent_ids, v_labels in dev_loader:
                        inputs = {
                            'input_ids': v_input_ids.to(device),
                            'attention_mask': v_attention_mask.to(device),
                            'labels': v_labels.to(device),
                        }
                        output = model(**inputs)
                        loss = output.loss
                        logits = output.logits.float().cpu()
                        pred = torch.argmax(logits, dim=1).cpu().tolist()

                        truths.extend(v_labels.cpu().tolist())
                        predictions.extend(pred)

                        valid_running_loss += loss.item()

                v_accu = compute_metrics((predictions, truths))["accuracy"]

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(dev_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss, v_accu))

                # checkpoint
                if best_valid_accu <= v_accu:
                    best_valid_accu = v_accu
                    save_checkpoint(model_file_path + '/' + model_name + ".pt", model, best_valid_accu)

    pbar.close()
    print("Finished Training")


# Evaluation Function

def do_eval(model, eval_loader, log_dir, model_name, device):
    y_pred = []
    y_True = []
    doc_keys = []
    ent_keys = []

    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, doc_ids, ent_ids, labels in eval_loader:
            inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device),
                'labels': labels.to(device),
            }
            output = model(**inputs)
            logits = output.logits.float().cpu()

            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
            y_True.extend(labels.cpu().tolist())
            doc_keys.extend(doc_ids)
            ent_keys.extend(ent_ids)

    print('Classification Report:')
    print(classification_report(y_True, y_pred, labels=[1, 0], digits=4))
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_True, y_pred, average='micro')

    with open(log_dir + "/eval-" + model_name + ".json", "w") as f_out:
        for d, s, y, t, in zip(doc_keys, ent_keys, y_pred, y_True):
            f_out.write(json.dumps({
                "doc_key": d,
                "ent_key": s,
                "predict": y,
                "truth": t
            }))
            f_out.write("\n")
        f_out.close()

    return precision, recall, f1_score


def do_predict(model, data_loader, log_dir, model_name, device):
    y_pred = []
    doc_keys = []
    ent_keys = []

    model.eval()
    p_bar = tqdm(range(data_loader.__len__()))
    with torch.no_grad():
        for input_ids, attention_mask, doc_ids, ent_ids in data_loader:
            inputs = {
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device),
            }
            output = model(**inputs)
            logits = output.logits.float().cpu()

            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
            doc_keys.extend(doc_ids)
            ent_keys.extend(ent_ids)
            p_bar.update(1)
    p_bar.close()

    with open(log_dir + "/pred-" + model_name + ".json", "w") as f_out:
        for d, s, y in zip(doc_keys, ent_keys, y_pred):
            f_out.write(json.dumps({
                "doc_key": d,
                "ent_key": s,
                "predict": y
            }))
            f_out.write("\n")
        f_out.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="../data/for_binary_classification", help="path of your data")
    parser.add_argument("-transformer_name", type=str, default="dmis-lab/biobert-v1.1",
                        help="model card of the pretrained bert model")
    parser.add_argument("-model_name", type=str, default="biobert-8", help="model name of your model")
    parser.add_argument("-learning_rate", type=float, default=15e-6, help="learning rate")
    parser.add_argument("-num_train_epochs", type=int, default=10, help="number of training epoch")
    parser.add_argument("-per_device_train_batch_size", type=int, default=8, help="batch size during training")
    parser.add_argument("-per_device_eval_batch_size", type=int, default=8, help="batch size during evaluation")
    parser.add_argument("-max_length", type=int, default=512, help="max length of tokens of input text")
    parser.add_argument("-do_negative_sampling", type=bool, default=True,
                        help="if set as false, all negative samples are used for training")
    parser.add_argument("-neg_ratio", type=int, default=8,
                        help="the number of negative samples is x times of positive samples")
    parser.add_argument("-training", type=bool, default=True, help="do training")
    parser.add_argument("-evaluation", type=bool, default=True, help="do evaluation")
    parser.add_argument("-prediction", type=bool, default=False, help="do prediction")
    args, _1 = parser.parse_known_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    args.n_gpu = torch.cuda.device_count()

    args.log_dir = "logs/" + args.model_name
    args.model_dir = "models/" + args.model_name

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    tokenizer = BertTokenizerFast.from_pretrained(args.transformer_name)
    config = BertConfig(args.transformer_name)

    def collate_fn(batch):
        input_ids = [f["input_ids"] + [tokenizer.pad_token_id] * (max_src_len - len(f["input_ids"])) for f in batch]
        attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_src_len - len(f["input_ids"])) for f in batch]
        doc_ids = [f["doc_key"] for f in batch]
        ent_ids = [f["ent_key"] for f in batch]
        labels = [f["label"] for f in batch]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, attention_mask, doc_ids, ent_ids, labels

    # Do training
    if args.training:
        model = BertBinaryClassifier(transformer_name=args.transformer_name, config=config, num_labels=2)
        model.to(args.device)

        print("Now the training is running...")
        # Load data

        all_data, max_src_len = load_bi_cls_data(args.data_dir, tokenizer, max_seq_length=512,
                                                 do_negative_sampling=args.do_negative_sampling,
                                                 neg_ratio=args.neg_ratio)

        train_loader = DataLoader(all_data["train"], batch_size=args.per_device_train_batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn, drop_last=False)

        dev_loader = DataLoader(all_data["dev"], batch_size=args.per_device_eval_batch_size, shuffle=False,
                                collate_fn=collate_fn, drop_last=False)

        # Set model
        model.to(args.device)

        for name, param in model.named_parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        print('# of Parameters: %d' % (
            sum([param.numel() if param.requires_grad is True else 0 for param in model.parameters()])))

        # Start training
        do_train(model=model, optimizer=optimizer, train_loader=train_loader,
                 dev_loader=dev_loader, eval_every=len(train_loader) // 2,
                 num_epochs=args.num_train_epochs,
                 model_file_path=args.model_dir, model_name=args.model_name, device=args.device)

        print('Finished training on all training data.')

    # Do evaluation
    if args.evaluation:
        best_model = BertBinaryClassifier(transformer_name=args.transformer_name, config=config, num_labels=2)
        load_checkpoint(args.model_dir + "/" + args.model_name + ".pt", best_model, device=args.device)
        best_model.to(args.device)
        all_data, max_src_len = load_bi_cls_data(args.data_dir, tokenizer,
                                                 max_seq_length=512, do_negative_sampling=args.do_negative_sampling,
                                                 test_only=True)

        eval_loader = DataLoader(all_data["test"], batch_size=args.per_device_eval_batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn, drop_last=False)
        precision, recall, f1_score = do_eval(best_model, eval_loader, args.log_dir, args.model_name, args.device)
        print(
            "Evaluation result on test set: Precision: {:.4f}, Recall: {:.4f}, Micro F1: {:.4f}.".format(precision,
                                                                                                         recall,
                                                                                                         f1_score))

    # Do prediction
    if args.prediction:
        def collate_pred_fn(batch):
            input_ids = [f["input_ids"] + [tokenizer.pad_token_id] * (max_src_len - len(f["input_ids"])) for f in batch]
            attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_src_len - len(f["input_ids"])) for f in batch]
            doc_ids = [f["doc_key"] for f in batch]
            ent_ids = [f["ent_key"] for f in batch]

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.float)
            return input_ids, attention_mask, doc_ids, ent_ids

        best_model = BertBinaryClassifier(transformer_name=args.transformer_name, config=config, num_labels=2)
        load_checkpoint(args.model_dir + "/" + args.model_name + ".pt", best_model, device=args.device)
        best_model.to(args.device)
        all_data, max_src_len = load_bi_cls_data_pred(args.data_dir, tokenizer, max_seq_length=512)

        eval_loader = DataLoader(all_data["pred"], batch_size=args.per_device_eval_batch_size,
                                 shuffle=False,
                                 collate_fn=collate_pred_fn, drop_last=False)
        do_predict(best_model, eval_loader, args.log_dir, args.model_name, args.device)


if __name__ == '__main__':
    main()
