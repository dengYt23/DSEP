import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
sys.path.append("..")
from rrgcn import DSEP
from DSE import get_historical_embeddings, THIS_DIR
from config import args
from rgcn import utils
from rgcn.utils import build_sub_graph
from rgcn.knowledge_graph import _read_triplets_as_list


def test(
    model: nn.Module, 
    history_list: list,
    test_list: list,
    all_ans_list: list, 
    all_ans_r_list: list, 
    model_name: str, 
    history_time: int, 
    mode: str
):
    """
    Test the model and evaluate its performance.

    Args:
        model (nn.Module): The model to be tested.
        history_list (list): List of historical snapshots.
        test_list (list): List of test snapshots.
        all_ans_list (list): List of all answers for entities.
        all_ans_r_list (list): List of all answers for relations.
        model_name (str): Path to the model checkpoint.
        history_time (int): History time offset for evaluation.
        mode (str): Mode of operation ('test' or otherwise).
    
    Returns:
        tuple: Evaluation metrics including MRR and Hits@K for entities and relations.
    """

    ranks_raw, ranks_filter = [], []
    ranks_raw_r, ranks_filter_r = [], []

    num_rels, num_nodes, use_cuda = model.num_rels, model.num_ents, model.use_cuda

    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    # Load model parameters
    if mode == "test":
        checkpoint = torch.load(model_name, map_location=device)
        print(f"Loaded model: {model_name}. Best epoch: {checkpoint['epoch']}")  # use best stat checkpoint
        print("\n" + "-"*10 + " Start Testing " + "-"*10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    his_graphs = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in history_list[-args.history_len:] + test_list]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [his_graphs[i] for i in range(time_idx, time_idx + args.history_len)]

        test_triples_input = torch.LongTensor(test_snap).to(device)
        inverse_test_triplets = test_triples_input[:, [2, 1, 0, 3]]
        inverse_test_triplets[:, 1] += num_rels
        all_triplets = torch.cat((test_triples_input, inverse_test_triplets))

        t = time_idx + history_time
        final_score, final_r_score = model.predict(
            history_glist,
            all_triplets,
            ent_ent_his_emb[t],
            ent_rel_his_emb[t],
            ent_rel_his_triplets_id[t],
            ent_ent_his_triplets_id[t],
        )

        rank_raw_r, rank_filter_r = utils.get_total_rank(
            all_triplets, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1
        )
        rank_raw, rank_filter = utils.get_total_rank(
            all_triplets, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0
        )

        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)

    mrr_raw, hit_result_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter, hit_result_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r, hit_result_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r, hit_result_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")

    if mode == "test" and args.write_output:
        hits = [1, 3, 10]
        os.makedirs(f"{THIS_DIR.parent}/outputs", exist_ok=True)
        with open(f"{THIS_DIR.parent}/outputs/{args.dataset}-outputs1.txt", 'a') as f:
            f.write(f"best epoch: {checkpoint['epoch']}\n")
            f.write(f"model_name: {model_name}\n")
            f.write("args: {}\n".format(args))
            f.write(f"mrr_raw: {mrr_raw}\n")
            for hit_i, hit in enumerate(hits):
                f.write(f"hits@{hit}: {hit_result_raw[hit_i]}\n")
            f.write(f"mrr_filter: {mrr_filter}\n")
            for hit_i, hit in enumerate(hits):
                f.write(f"hits@{hit}: {hit_result_filter[hit_i]}\n")
            f.write(f"mrr_raw_r: {mrr_raw_r}\n")
            for hit_i, hit in enumerate(hits):
                f.write(f"hits@{hit}: {hit_result_raw_r[hit_i]}\n")
            f.write(f"mrr_filter_r: {mrr_filter_r}\n")
            for hit_i, hit in enumerate(hits):
                f.write(f"hits@{hit}: {hit_result_filter_r[hit_i]}\n")
            f.write("\n")

    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r


def run_experiment(
    args: argparse.Namespace,  
    history_len: int = None  
): 
    """
    Train the model and evaluate performance.

    Args:
        args (Namespace): Configuration arguments for training, including model parameters, hyperparameters, dataset paths, GPU settings, etc.
        history_len (int, optional): Length of history for training and testing. If provided, overrides the default history length in args.

    Returns:
        tuple: Evaluation metrics including MRR and Hits@K for entities and relations.
    """
    if history_len:
        args.train_history_len = history_len
        args.test_history_len = history_len

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list, train_times = utils.split_by_time(data.train)  # Split into snapshots
    valid_list, _ = utils.split_by_time(data.valid)
    test_list, _ = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_times = len(train_list) + len(valid_list) + len(test_list) + (1 if args.dataset == "ICEWS14" else 0)
    time_interval = train_times[1] - train_times[0]

    print(f"Number of time steps: {num_times}, Time interval: {time_interval}")
    history_val_time = len(train_list)  # valid list start time
    history_test_time = len(train_list) + len(valid_list)  # test list start time

    # Load answers for evaluation
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, True)
    
    model_name = f"{args.plm}_gl_rate_{args.history_rate}-{args.dataset}-{args.gcn}-{args.decoder}-ly{args.n_layers}-his{args.history_len}-num-k{args.num_k}"
    os.makedirs(f'{THIS_DIR.parent}/models', exist_ok=True)
    model_state_file = os.path.join(THIS_DIR.parent, 'models/', model_name)

    print(f"Model save path: {model_state_file}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

    all_garphs = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in train_list]

    # Load static graph
    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list(f"{THIS_DIR.parent}/data/{args.dataset}/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] +=  num_nodes  # Adjust node IDs
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().to(device)
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    model = DSEP(
        args.gcn, args.decoder, num_nodes, num_rels, num_static_rels, num_words, time_interval, args.n_hidden,
        args.opn, args.history_rate, num_bases=args.n_bases, num_basis=args.n_basis, num_hidden_layers=args.n_layers,
        dropout=args.dropout, self_loop=args.self_loop, skip_connect=args.skip_connect, layer_norm=args.layer_norm,
        input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout, feat_dropout=args.feat_dropout, 
        static_weight=args.weight, discount=args.discount, angle=args.angle, use_static=args.add_static_graph, use_cuda=use_cuda,
        gpu=args.gpu, analysis=args.run_analysis, hidden_size=embedding_dim, ent_emb=ent_embs, rel_emb=rel_embs, 
        static_graph=static_graph
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        results = test(model, train_list + valid_list, test_list, all_ans_list_test, all_ans_list_r_test, model_state_file,
                       history_test_time, mode="test")
    elif args.test:
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        
        best_mrr = 0
        best_epoch = 0

        for epoch in range(1, 1 + args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            # Shuffle training data indices
            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for t in tqdm(idx):
                if t == 0: 
                    continue

                current_triplets = torch.tensor(train_list[t], dtype=torch.long, device=device)

                inverse_triples = current_triplets[:, [2, 1, 0, 3]]
                inverse_triples[:, 1] += num_rels
                all_triplets = torch.cat([current_triplets, inverse_triples])

                history_glist = [all_garphs[i] for i in range(max(0, t - args.history_len), t)]

                loss_e, loss_r, loss_static = model.get_loss(
                    history_glist, all_triplets, ent_ent_his_emb[t], ent_rel_his_emb[t],
                    ent_rel_his_triplets_id[t], ent_ent_his_triplets_id[t]
                )
                loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r + loss_static
                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()
            
            print(f"Epoch {epoch:04d}/{args.n_epochs:04d} | Avg Loss: {np.mean(losses):.4f} | "
                  f"Entity-Relation-Static Loss: {np.mean(losses_e):.4f}-{np.mean(losses_r):.4f}-{np.mean(losses_static):.4f} "
                  f"Best MRR: {best_mrr:.4f} Best Epoch: {best_epoch:04d} Model: {model_name}")

            # validation
            if epoch % args.evaluate_every == 0:
                results = test(model, train_list, valid_list, all_ans_list_valid, all_ans_list_r_valid, model_state_file,
                               history_val_time, mode="train")
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if results[0] > best_mrr:
                        best_mrr = results[0]
                        best_epoch = epoch
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if results[2] > best_mrr:
                        best_mrr = results[2]
                        best_epoch = epoch
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            if epoch >= args.n_epochs:
                break

        results = test(model, train_list + valid_list, test_list, all_ans_list_test, all_ans_list_r_test, model_state_file, 
                       history_test_time, mode="test")


if __name__ == '__main__':
    # Load historical embeddings
    ent_embs, rel_embs, ent_ent_his_emb, ent_rel_his_emb, ent_ent_his_triplets_id, ent_rel_his_triplets_id, new_entity_ids  = get_historical_embeddings(
        args.dataset, args.num_k, plm=args.plm, model_type=args.model_type, batch_size=args.batch_size, gpu=args.gpu, save=True
        )
                
    embedding_dim = ent_embs.shape[1]
    run_experiment(args)


