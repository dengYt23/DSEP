import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5Model, GPT2Tokenizer, GPT2Model, LlamaTokenizer, LlamaModel, AutoTokenizer
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()


class DataPreprocessor:
    def __init__(self, dataset, num_k, plm='bert-large-based'):
        self.dataset = dataset
        self.num_k = num_k
        self.plm = plm
        
    def _build_path(self, *args):
        """
        Construct a file path based on dataset and model type.
        """

        return os.path.join(THIS_DIR.parent, 'data', self.dataset, 'text_emb', self.plm, *args)
    
    def get_paths(self):
        """
        Get file paths for saving embeddings based on the dataset and model.
        """

        base_path = self._build_path()
        path = base_path
        entities_emb_path = self._build_path('entities_emb.pt')
        relations_emb_path = self._build_path('relations_emb.pt')
        ent_ent_his_emb_path = self._build_path(f'ent_ent_his_emb_num{self.num_k}.pt')
        ent_rel_his_emb_path = self._build_path(f'ent_rel_his_emb_num{self.num_k}.pt')
        return {
            'path': path,
            'entities_emb_path': entities_emb_path, 
            'relations_emb_path': relations_emb_path, 
            'ent_ent_his_emb_path': ent_ent_his_emb_path, 
            'ent_rel_his_emb_path': ent_rel_his_emb_path
            }

    def load_id_data(self, fileNames=['train.txt', 'valid.txt', 'test.txt']):
        """
        Load data from the dataset files.
        """

        all_data = defaultdict(list)
        for fileName in fileNames:
            with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, fileName), 'r') as fr:
                # some datasets have 5 columns(s, r, o, [start_time, end_time]), some have 4(s, r, o, time)
                for line in fr:
                    parts = list(map(int, line.split()))
                    head, rel, tail, time = parts[:4]
                    if self.dataset == 'ICEWS14': # ICEWS14 starts from 1
                        time -= 1
                    all_data.setdefault(time, []).append([head, rel, tail])
        return all_data

    def get_id2ent_id2rel(self):
        """
        Load mappings from ID to entity and relation names.
        """

        if self.dataset == 'GDELT':
            id2ent = self._load_id2ent_GDELT()
            id2rel = self._load_id2rel_GDELT()
        else:
            id2ent, id2rel = self._load_id2ent_id2rel_default()

        return id2ent, id2rel

    def _load_id2ent_GDELT(self):
        """
        Load entity mappings specifically for the GDELT dataset.
        """

        with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'entity2id.txt'), 'r') as f:
            return {int(line.split('\t')[1]): line.split('\t')[0].split('(')[0].strip() for line in f}

    def _load_id2rel_GDELT(self):
        """
        Load relation mappings specifically for the GDELT dataset.
        """

        with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'relation2id.txt'), 'r') as f1, \
             open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'CAMEO-eventcodes.txt'), 'r') as f2:
            id2cameoCode = {int(line.split('\t')[1]): line.split('\t')[0] for line in f1}
            cameoCode2rel = {line.split('\t')[0]: line.split('\t')[1].strip() for line in f2}
            return {id_: cameoCode2rel[id2cameoCode[id_]] for id_ in id2cameoCode.keys()}

    def _load_id2ent_id2rel_default(self):
        """
        Load entity and relation mappings for datasets other than GDELT.
        """

        with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'entity2id.txt'), 'r') as f:
            id2ent = {int(line.split('\t')[1]): line.split('\t')[0] for line in f}
        with open(os.path.join(THIS_DIR.parent, 'data', self.dataset, 'relation2id.txt'), 'r') as f:
            id2rel = {int(line.split('\t')[1]): line.split('\t')[0] for line in f}
        return id2ent, id2rel

    def generate_historical_triplets(self):
        """
        Generate historical triplets for each time step, encoding them into 
        both sentence-based triplets and ID-based triplets for future usage.
        """

        all_data = self.load_id_data()
        id2ent, id2rel = self.get_id2ent_id2rel()
        rel_nums = len(id2rel)
        times = list(all_data.keys())
        
        # Determine time interval (for ICEWS or GDELT datasets)
        # Time interval: GDELT: 15mins ICEWS: 1day ---> 1day = 15mins *4 * 24
        time_interval = times[1] - times[0] if self.dataset != 'GDELT' else (times[1] - times[0]) * 24 * 4
        
        # Initialize containers for triplets and historical triplets
        ent_ent_triplets = defaultdict(list)
        ent_rel_triplets = defaultdict(list)
        ent_ent_his_triplets = defaultdict(list)
        ent_rel_his_triplets = defaultdict(list)

        ent_ent_triplets_id = defaultdict(list)
        ent_rel_triplets_id = defaultdict(list)
        ent_ent_his_triplets_id = defaultdict(list)
        ent_rel_his_triplets_id = defaultdict(list)

        seen_entity_ids = set()  # Track all seen s_id
        new_entity_ids = defaultdict(list)
        
        for idx, t in enumerate(tqdm(all_data.keys(), desc="Generating historical triplets")):
            # Process current time step triplets
            train_new_data = torch.from_numpy(np.array(all_data[t]))

            # Generate inverse triplets
            inverse_train_data = train_new_data[:, [2, 1, 0]]
            inverse_train_data[:, 1] = inverse_train_data[:, 1] + rel_nums
            train_new_data = torch.cat([train_new_data, inverse_train_data])

            t //= time_interval
            date = self.convert_to_date(t)

            ent_ent_triplets_t, ent_rel_triplets_t = [], []
            ent_ent_update, ent_rel_update = defaultdict(list), defaultdict(list)

            ent_ent_triplets_t_id, ent_rel_triplets_t_id = [], []
            ent_ent_update_id, ent_rel_update_id = defaultdict(list), defaultdict(list)

            s_ids = train_new_data[:, 0].unique().tolist()
            unseen_entity_ids = set(s_ids) - seen_entity_ids

            # Process each triplet in the current batch
            for k, (s_id, r_id, o_id) in enumerate(train_new_data.tolist()):
                if r_id < rel_nums:
                    # Process normal triplet
                    s, r, o = id2ent[s_id], id2rel[r_id], id2ent[o_id]
                    sentence = f"{s} {r} {o} on {date}"
                    ent_ent_triplets_t.append(ent_ent_triplets.get((s, o), []) + [f"{s} ? {o} on {date}"])
                    ent_ent_update[(s, o)].append(sentence)
                    ent_rel_triplets_t.append(ent_rel_triplets.get((s, r), []) + [f"{s} {r} ? on {date}"])
                    ent_rel_update[(s, r)].append(sentence)
                else:
                    # Process inverse triplet
                    s, r, o = id2ent[s_id], id2rel[r_id - rel_nums], id2ent[o_id]
                    sentence = f"{o} {r} {s} on {date}"
                    ent_ent_triplets_t.append(ent_ent_triplets.get((s, o), []) + [f"{o} ? {s} on {date}"])
                    ent_ent_update[(s, o)].append(sentence)
                    ent_rel_triplets_t.append(ent_rel_triplets.get((s, r), []) + [f"? {r} {s} on {date}"])
                    ent_rel_update[(s, r)].append(sentence)

                # store the id of the triplets
                ent_ent_triplets_t_id.append(ent_ent_triplets_id.get((s_id, o_id), []))
                ent_ent_update_id[(s_id, o_id)].append(r_id)
                ent_rel_triplets_t_id.append(ent_rel_triplets_id.get((s_id, r_id), []))
                ent_rel_update_id[(s_id, r_id)].append(o_id)
                
                if s_id in unseen_entity_ids:
                    new_entity_ids[t].append(k)
            
            # Save historical triplets for this time step
            ent_ent_his_triplets[idx] = copy.deepcopy(ent_ent_triplets_t)
            ent_rel_his_triplets[idx] = copy.deepcopy(ent_rel_triplets_t)

            ent_ent_his_triplets_id[idx] = copy.deepcopy(ent_ent_triplets_t_id)
            ent_rel_his_triplets_id[idx] = copy.deepcopy(ent_rel_triplets_t_id)

            # Update historical triplets for future steps
            self._update_historical_triplets(ent_ent_triplets, ent_ent_update)
            self._update_historical_triplets(ent_rel_triplets, ent_rel_update)

            self._update_triplet_ids(ent_ent_triplets_id, ent_ent_update_id)
            self._update_triplet_ids(ent_rel_triplets_id, ent_rel_update_id)

        return ent_ent_his_triplets, ent_rel_his_triplets, ent_ent_his_triplets_id, ent_rel_his_triplets_id, new_entity_ids

    def _update_historical_triplets(self, triplets, updates):
        """
        Update historical triplets with new sentences, ensuring no more than `num_k` triplets are stored.
        """

        for key, sentences in updates.items():
            triplets[key].extend(sentences)
            while len(triplets[key]) > self.num_k:
                triplets[key].pop(0)

    def _update_triplet_ids(self, triplets_id, updates_id):
        """
        Update triplet IDs, ensuring no duplicate IDs are stored.
        """

        for key, ids in updates_id.items():
            for r_id in ids:
                if r_id not in triplets_id[key]:
                    triplets_id[key].append(r_id)
        
    def get_local_entity(self, his_triplets, current_triplets):
        """
        Get the local entities connected to the current triplet, based on historical triplets.
    """
        local_entities = []
        his_entities = defaultdict(set)

        # Create a mapping from subject to object and relation
        for his_triplet in his_triplets:
            his_triplet = torch.from_numpy(his_triplet)
            inverse_his_triplet = his_triplet[:, [2, 1, 0, 3]]
            all_triplets = torch.cat([his_triplet, inverse_his_triplet], dim=0)
            
            for triplet in all_triplets:
                s1, r1, o1 = triplet[:3].tolist()
                his_entities[s1].add((o1, r1))
            
        # Find local entities related to the current triplet
        for triplet in current_triplets:
            s, r = triplet[:2].tolist()
            query = set()

            if s in his_entities:
                for o1, r1 in his_entities[s]:
                    query.add(o1)
                    if r1 == r:
                        query.update(o2 for o2, _ in his_entities[o1])

            local_entities.append(query)
        
        return local_entities
    
    def convert_to_date(self, number):
        """
        Convert the number to date string. For example, 0 -> '2014-01-01'.
        """

        if self.dataset in ['GDELT', 'ICEWS18']:
            self.year = 2018
        elif self.dataset in ['ICEWS05-15']:
            self.year = 2005
        elif self.dataset in ['ICEWS14s', 'ICEWS14']:
            self.year = 2014
        else:
            raise ValueError('Unsupported dataset')
        
        base_date = datetime(self.year, 1, 1)
        delta = timedelta(days=number)
        target_date = base_date + delta
        return target_date.strftime("%Y-%m-%d")


class DSEncoder(nn.Module):
    def __init__(self, paths, plm='bert-large-cased', model_type='bert', batch_size=32, gpu=-1, save=False):
        super(DSEncoder, self).__init__()

        self.device = torch.device(f'cuda:{gpu}' if gpu >=0 and torch.cuda.is_available() else 'cpu')
        self.plm = plm
        self.model_type = model_type
        self.batch_size = batch_size
        self.save = save

        # Load file paths
        self.path = paths['path']
        self.entities_emb_path = paths['entities_emb_path']
        self.relations_emb_path = paths['relations_emb_path']
        self.ent_ent_his_emb_path = paths['ent_ent_his_emb_path']
        self.ent_rel_his_emb_path = paths['ent_rel_his_emb_path']

        self._load_pretrained_model()  # Load the specified pre-trained model


    def _load_pretrained_model(self):
        """
        Load the specified pre-trained model (BERT, T5, GPT-2, etc.) and its tokenizer.
        """

        model_path = os.path.join(THIS_DIR.parent, 'plm', self.model_type, self.plm)

        print(f'\nLoading pretrained model from {model_path}...\n')

        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path).to(self.device)
            self.max_tokens = self.model.config.max_position_embeddings  # max token limit
            self.hidden_size = self.model.config.hidden_size
        elif self.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5Model.from_pretrained(model_path).to(self.device)
            self.max_tokens = self.tokenizer.model_max_length
            self.hidden_size = self.model.config.d_model
        elif self.model_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token for GPT-2
            self.model = GPT2Model.from_pretrained(model_path).to(self.device)
            self.max_tokens = self.model.config.max_position_embeddings
            self.hidden_size = self.model.config.hidden_size
        elif self.model_type == 'llama':
            model_path = os.path.join(THIS_DIR.parent, 'meta-llama/Meta-Llama-3.1-8B')
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token for GPT-2
            self.model = LlamaModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                )
            self.hidden_size = self.model.config.hidden_size
        else:
            """
            Add more models here.
            """
            raise ValueError('Unsupported model type')

    def _get_sentence_embedding(self, sentences):
        """
        Generate sentence embeddings using the pre-trained model.
        """

        with torch.no_grad():
            inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            if self.model_type in ['gpt2', 'llama']:
                # GPT-2 does not have a [CLS] token, use mean of hidden states
                return torch.mean(outputs.last_hidden_state, dim=1)
            return outputs.last_hidden_state[:, 0, :]

    def initial_word_embedding(self, entities=None, relations=None):
        """
        Encode and save embeddings for entities and relations.
        """

        if os.path.exists(self.entities_emb_path) and os.path.exists(self.relations_emb_path):
            print('\nLoading existing entity and relation embeddings...\n')
            entities_embedding = torch.load(self.entities_emb_path).to(self.device)
            relations_embedding = torch.load(self.relations_emb_path).to(self.device)
            return entities_embedding, relations_embedding
        
        entities_embedding = torch.zeros(len(entities), self.hidden_size).to(self.device)
        relations_embedding = torch.zeros(len(relations), self.hidden_size).to(self.device)

        print('\n' + '-'*10 + 'Encoding entities and relations' + '-'*10)
        
        for idx, entity in tqdm(enumerate(entities), desc='Encoding entities'):
            entities_embedding[idx] = self._get_sentence_embedding(entity)
        
        for idx, relation in tqdm(enumerate(relations), desc='Encoding relations'):
            relations_embedding[idx] = self._get_sentence_embedding(relation)
        
        if self.save:
            os.makedirs(self.path, exist_ok=True)
            torch.save(entities_embedding.cpu(),self.entities_emb_path)
            print('The encoded entities is saved in:', self.entities_emb_path)

            torch.save(relations_embedding.cpu(), self.relations_emb_path)
            print('The encoded relations is saved in:', self.relations_emb_path)

        return entities_embedding, relations_embedding

    def encode_his_triplets(self, his_triplets):
        """
        Encode historical triplets into embeddings.
        """

        embeddings = defaultdict(dict)

        for t, his_triplet in tqdm(his_triplets.items()):
            sentences = self._build_sentences_from_triplets(his_triplet)
            embeddings[t] = self._embed_sentences_in_batches(sentences)
        
        return embeddings

    def _build_sentences_from_triplets(self, his_triplet):
        """
        Helper function to create sentences from historical triplets.
        """

        if self.model_type == 'bert':
            return [' [SEP] '.join(triplet) for triplet in his_triplet]
        elif self.model_type in ['t5', 'gpt2', 'llama']:
            return ['. '.join(triplet) for triplet in his_triplet]

    def _embed_sentences_in_batches(self, sentences):
        """
        Helper function to generate embeddings for sentences in batches.
        """

        embeddings_t = torch.zeros(len(sentences), self.hidden_size).to(self.device)
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            batch_embeddings = self._get_sentence_embedding(batch_sentences)
            embeddings_t[i:i + self.batch_size, :] = batch_embeddings
        return embeddings_t
    
    def encode(self, ent_ent_his_triplets, ent_rel_his_triplets):
        """
        Encode historical triplets for both entity-entity and entity-relation pairs.
        """

        if os.path.exists(self.ent_ent_his_emb_path) and os.path.exists(self.ent_rel_his_emb_path):
            print(f'\nLoading existing entity-entity and entity-relation historical embeddings\n')
            ent_ent_his_embeddings = torch.load(self.ent_ent_his_emb_path)
            ent_rel_his_embeddings = torch.load(self.ent_rel_his_emb_path)
            return ent_ent_his_embeddings, ent_rel_his_embeddings
        
        ent_ent_his_embeddings = self.encode_his_triplets(ent_ent_his_triplets)
        ent_rel_his_embeddings = self.encode_his_triplets(ent_rel_his_triplets)

        if self.save:
            self._save_his_embeddings(ent_ent_his_embeddings, self.ent_ent_his_emb_path, 'ent_ent_his_embeddings')
            self._save_his_embeddings(ent_rel_his_embeddings, self.ent_rel_his_emb_path, 'ent_rel_his_embeddings')
        
        return ent_ent_his_embeddings, ent_rel_his_embeddings

    def _save_his_embeddings(self, embeddings, path, name):
        """
        Helper function to save history embeddings.
        """

        os.makedirs(self.path, exist_ok=True)
        for t, e in embeddings.items():
            embeddings[t] = e.cpu()
        torch.save(embeddings, path)
        print(f'{name} are saved in:\n', path)


def get_historical_embeddings(dataset, num_k, plm='bert-large-cased', model_type='bert', batch_size=32, gpu=-1, save=False):
    """
    Generate embeddings for historical triplets.

    Args:
        dataset (str): The name of the dataset to use.
        num_k (int): The number of triplets to store for each entity-entity and entity-relation pair.
        plm (str, optional): The pre-trained language model to use (default is 'bert-large-cased').
        model_type (str, optional): The type of pre-trained model ('bert', 't5', etc.).
        batch_size (int, optional): The batch size for batch embedding generation (default is 32).
        gpu (int, optional): The GPU index to use (default is -1 for CPU).
        save (bool, optional): Whether to save the embeddings to disk (default is False).

    Returns:
        Tuple: A tuple containing the following elements:
            - entities_embedding (Tensor)
            - relations_embedding (Tensor)
            - ent_ent_his_embeddings (Tensor): Historical embeddings for entity-entity relations.
            - ent_rel_his_embeddings (Tensor): Historical embeddings for entity-relation relations.
            - ent_ent_his_triplets_id (List): IDs for entity-entity historical triplets.
            - ent_rel_his_triplets_id (List): IDs for entity-relation historical triplets.
    """

    preprocessor = DataPreprocessor(dataset, num_k, plm)
    
    # Generate file paths and historical triplets
    paths = preprocessor.get_paths()
    ent_ent_his_triplets, ent_rel_his_triplets, ent_ent_his_triplets_id, ent_rel_his_triplets_id, new_entity_ids = preprocessor.generate_historical_triplets()
    id2ent, id2rel = preprocessor.get_id2ent_id2rel()
    entities, relations = list(id2ent.values()), list(id2rel.values())

    encoder = DSEncoder(paths, plm, model_type, batch_size, gpu, save)
    
    # Generate initial embeddings
    entities_embedding, relations_embedding = encoder.initial_word_embedding(entities, relations)
    
    # Generate historical embeddings
    ent_ent_his_embeddings, ent_rel_his_embeddings = encoder.encode(ent_ent_his_triplets, ent_rel_his_triplets)
    
    return entities_embedding, relations_embedding, ent_ent_his_embeddings, ent_rel_his_embeddings, ent_ent_his_triplets_id, ent_rel_his_triplets_id, new_entity_ids


