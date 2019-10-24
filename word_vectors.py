from aux import *
import numpy as np
import collections

def get_sentences(words,ds,fileid):
    text = get_text(ds,fileid)
    print(ds, fileid, len(words))
    T = []
    L = []
    for sent in text.iter('sentence'):
        t = []
        l = []
        for w in sent:
            t.append(w.text)
            l.append(w.get('lemma').lower())
        T.append(t)
        L.append(l)
    I = []
    idx = 0
    for l in L:
        i = []
        for wi, w in enumerate(l):
            if w == words[idx]:
                i.append(wi)
                idx += 1
                if idx >= len(words):
                    break
        I.append(i)
    length = max(len(l) for l in L)
    lengths = []
    TT = []
    for t in L:
        lt = len(t)
        lengths.append(lt)
        t = t + [""] * (length - lt)
        TT.append(t)
    return T, I

def get_bert(Tokens, Indices, mdl, layer):
    tot=0
    model_class = BertModel
    tokenizer_class = BertTokenizer
    pretrained_weights = mdl
    Mat = []
    with open('bert_text.txt', 'w') as f:
        for t in Tokens:
            f.write(' '.join(t) + '\n')
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_file", default='bert_text.txt', type=str, required=False)
    parser.add_argument("--output_file", default='output_bert.txt', type=str, required=False)
    parser.add_argument("--bert_model", default=mdl, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    # if 'uncased' in mdl:
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--mode", default='client', type=str)
    parser.add_argument("--output_hidden_states", default='True', type=bool)
    parser.add_argument("--port", default=51940, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=250, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    args = parser.parse_args()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))
    layer_indexes = [int(x) for x in args.layers.split(",")]
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    examples = read_examples(args.input_file)
    features = convert_examples_to_features(examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
    for input_ids, input_mask, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        all_hidden_states, all_attentions = model(input_ids, token_type_ids=None, attention_mask=input_mask)[-2:]
        for b, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            unique_id = int(feature.unique_id)
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_out_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = all_hidden_states[int(layer_index)].detach().cpu().numpy()
                    layer_output = layer_output[b]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [round(x.item(), 6) for x in layer_output[i]]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = token
                out_features["layers"] = all_layers
                all_out_features.append(out_features)
            if 'uncased' in mdl.lower():
                ww = [Tokens[b][i].lower() for i in Indices[b]]
            else:
                ww = [Tokens[b][i] for i in Indices[b]]
            ww_bert = [out_features['token'] for out_features in all_out_features]
            Idx = []
            print('SENTENCE', b)
            print(ww)
            print(ww_bert)
            if len(ww) > 0:
                idx_bert = 0
                for i, www in enumerate(ww):
                    www = www.replace('(', '')
                    www = www.replace(')', '')
                    for w in ww_bert[idx_bert:]:
                        if w in ['[SEP]', '[CLS]', '[UNK]']:
                            idx_bert += 1
                            continue
                        if w == www:
                            Idx.append([idx_bert])
                            idx_bert += 1
                            break
                        if ' ' in www:
                            if www.split(' ')[0] == w:
                                Idx.append([idx_bert])
                                idx_bert += 1
                                for w_ in www.split(' ')[1:]:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                while ww_bert[idx_bert].startswith('##') and ww_bert[idx_bert].strip('#') in www:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                break
                        if '\'' in www:
                            if www.split('\'')[0] == w:
                                Idx.append([idx_bert])
                                idx_bert += 1
                                break
                        if '-' in www:
                            if www.split('-')[0] == w:
                                Idx.append([idx_bert])
                                idx_bert += 1
                                for w_ in ['-'] + www.split('-')[1:]:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                while ww_bert[idx_bert].startswith('##') and ww_bert[idx_bert].strip('#') in www:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                break
                        if '/' in www:
                            if www.split('/')[0] == w:
                                Idx.append([idx_bert])
                                idx_bert += 1
                                for w_ in ['/'] + www.split('/')[1:]:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                while ww_bert[idx_bert].startswith('##') and ww_bert[idx_bert].strip('#') in www:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                break
                        if '.' == www[-1]:
                            if www.split('.')[0] == w:
                                Idx.append([idx_bert])
                                idx_bert += 1
                                www_list = [x for x in www.split('.')[1:] if x != '']
                                for w_ in www_list + ['.']:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                while ww_bert[idx_bert].startswith('##') and ww_bert[idx_bert].strip('#') in www:
                                    Idx[-1].append(idx_bert)
                                    idx_bert += 1
                                break

                        if www.startswith(w) and ww_bert[idx_bert + 1].startswith('##') and www not in ww_bert[idx_bert:]:
                            Idx.append([idx_bert])
                            idx_bert += 1
                            continue
                        if len(Idx) > 0:
                            if w.startswith('##') and Idx[-1][-1] == idx_bert - 1:
                                Idx[-1].append(idx_bert)
                                idx_bert += 1
                                if ww_bert[idx_bert].startswith('##'):
                                    continue
                                else:
                                    break
                        idx_bert += 1
                print(ww)
                print(Idx)
                for ix, i in enumerate(Idx):
                    for ii in i:
                        print(ww[ix], all_out_features[ii]['token'])
                        try:
                            all_out_features[ii]['token'].strip('#') in ww[ix]
                        except:
                            print('No')
                            assert all_out_features[ii]['token'].strip('#') in ww[ix]
                for idx in Idx:
                    tot+=1
                    if layer == 'sum':
                        vec1 = torch.mean(torch.tensor([all_out_features[iidx]['layers'][0]['values'] for iidx in idx]),0)
                        vec2 = torch.mean(torch.tensor([all_out_features[iidx]['layers'][1]['values'] for iidx in idx]),0)
                        vec3 = torch.mean(torch.tensor([all_out_features[iidx]['layers'][2]['values'] for iidx in idx]),0)
                        vec4 = torch.mean(torch.tensor([all_out_features[iidx]['layers'][3]['values'] for iidx in idx]),0)
                        vec = vec1 + vec2 + vec3 + vec4
                    if layer == 'conc':
                        vec = torch.mean(torch.tensor([all_out_features[iidx]['layers'][0]['values'] +
                                                        all_out_features[iidx]['layers'][1]['values'] +
                                                        all_out_features[iidx]['layers'][2]['values'] +
                                                        all_out_features[iidx]['layers'][3]['values'] for iidx in
                                                        idx]), 0)
                    if layer not in ['sum','conc']:
                        vec = torch.mean(torch.tensor([all_out_features[iidx]['layers'][int(layer)-1]['values'] for iidx in idx]),
                                          0)
                    Mat.append(vec.tolist())
    print(tot)
    print(len(Mat))
    return Mat


## ----------------

def get_bert_vectors(words,ds,fileid,word_embedding,use_svd,save_matrices):
    Tokens,Indices = get_sentences(words,ds,fileid)
    mdl = get_model(word_embedding)
    layer = word_embedding.split('-')[-1]
    Mat = get_bert(Tokens, Indices, mdl, layer)
    print(len(Mat))
    fileid = fileid.split('.')[0]
    fileid = fileid.split('-')[-1]
    if save_matrices == True:
        path = 'vectors/' + ds + '/word/' + mdl + '-' + layer
        create_directory(path)
        writeMat(Mat, path + '/A-' + fileid)
    if use_svd == True:
        Mat = np.array(Mat)
        Mat = remove_pc(Mat, npc=1)
        if save_matrices == True:
            writeMat(Mat, path + '/A-' + fileid + '-svd')
