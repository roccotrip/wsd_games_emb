from aux import *

def get_Idx(model,T,I,all_input_ids,eval_dataloader,device,features,layer_indexes,mdl):
    #w_idx = all_input_ids.numpy()[all_input_ids.numpy() > 0].tolist()
    #word_embedd = model.embeddings.word_embeddings.weight.data[w_idx]

    for input_ids, input_mask, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        all_encoder_layers = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        for b, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            unique_id = int(feature.unique_id)
            # feature = unique_id_to_feature[unique_id]
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_out_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = all_encoder_layers[int(layer_index)]#.detach().cpu().numpy()
                    layer_output = layer_output[b]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    #layers["values"] = [round(x.item(), 6) for x in layer_output[i]]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = token
                out_features["layers"] = all_layers
                all_out_features.append(out_features)
            if 'uncased' in mdl:
                ww = [T[b][i].lower() for i in I[b]]
            else:
                ww = [T[b][i] for i in I[b]]
            ww_bert = [out_features['token'] for out_features in all_out_features]
            idx0 = 0
            Idx = []
            f = 0
            print(ww)
            print(ww_bert)
            if len(ww) > 0:
                idx_bert=0
                for i, www in enumerate(ww):
                    www=www.replace('(','')
                    www=www.replace(')','')
                    if www == 'key' and i == 0:
                        print('o')
                    for w in ww_bert[idx_bert:]:
                        if w in ['[SEP]','[CLS]','[UNK]']: #or (w in stopset and w not in ww and ww_bert[idx_bert+1].startswith('#')==False and ' ' in www == False):
                            idx_bert+=1
                            continue
                        if w == www:
                            Idx.append([idx_bert])
                            idx_bert+=1
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

                        if www.startswith(w) and ww_bert[idx_bert+1].startswith('##') and www not in ww_bert[idx_bert:]:
                            Idx.append([idx_bert])
                            idx_bert += 1
                            continue
                        if len(Idx) > 0:
                            if w.startswith('##') and Idx[-1][-1] == idx_bert-1:
                                Idx[-1].append(idx_bert)
                                idx_bert += 1
                                if ww_bert[idx_bert].startswith('##'):
                                    continue
                                else:
                                    break
                        idx_bert+=1
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
                try:
                    assert len(Idx) == len(ww)
                except:
                     print('o')
    return Idx

def compute_heads_importance(args, model, eval_dataloader, tokens, Idx, compute_entropy=True, compute_importance=False, head_mask=None):
    """ Example on how to use model outputs to compute:
        - head attention entropy (activated by setting output_attentions=True when we created the model
        - head importance scores according to http://arxiv.org/abs/1905.10650
            (activated by setting keep_multihead_output=True when we created the model)
    """

    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)
    preds = []
    labels = []
    tot_tokens = 0.0
    plt.close('all')
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids = batch
        label_ids = segment_ids

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)

        all_attentions = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, head_mask=head_mask)
        all_attentions = all_attentions[3]

        ntokens = int(input_mask.float().detach().sum().data)
        att_probs = head_importance = torch.zeros(n_layers,len(Idx),len(Idx))

        # for layer, attn in enumerate(all_attentions):
        #     plt.figure(layer)
        #     seaborn.heatmap(attn[0].sum(dim=0)[0:ntokens, 0:ntokens].detach().numpy(),xticklabels=tokens, yticklabels=tokens, annot=True)
        #     plt.show()


        if compute_entropy:
            # Update head attention entropy
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * input_mask.float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        layer_entropies = attn_entropy.sum(dim=1)

        labels_att = []
        for idx in Idx:
            labels_att.append(''.join([tokens[iidx] for iidx in idx]))

        for layer, attn in enumerate(all_attentions):
            att = []
            att_all = attn[0].sum(dim=0)
            att_all = att_all / att_all.sum(1, keepdim=True)
            att_all = att_all.detach().numpy()
            for idx in Idx:
                att_ = torch.sum(torch.tensor([att_all[iidx] for iidx in idx]), 0).numpy().tolist()[:ntokens]
                att2 = []
                for ii,idx_ in enumerate(Idx):
                    att2.append(sum([att_[iidx] for iidx in idx_]))
                att.append(att2)
            att_probs[layer] = torch.tensor(att)
            plot_all = False
            plot = False
            if plot == True:
                plt.figure(layer)
                seaborn.heatmap(att,xticklabels=labels_att, yticklabels=labels_att, annot=False)
                seaborn.set(font_scale=.75)
                plt.show()
            if plot_all == True:
                plt.figure(layer)
                seaborn.heatmap(att_all[0:len(tokens),0:len(tokens)],xticklabels=tokens, yticklabels=tokens, annot=False)
                seaborn.set(font_scale=.75)
                plt.show()

        # if compute_importance:
        #     # Update head importance scores with regards to our loss
        #     # First, backpropagate to populate the gradients
        #     if args.output_mode == "classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, args.num_labels), label_ids.view(-1))
        #     elif args.output_mode == "regression":
        #         loss_fct = MSELoss()
        #         loss = loss_fct(logits.view(-1), label_ids.view(-1))
        #     loss.backward()
        #     # Second, compute importance scores according to http://arxiv.org/abs/1905.10650
        #     multihead_outputs = model.bert.get_multihead_outputs()
        #     for layer, mh_layer_output in enumerate(multihead_outputs):
        #         dot = torch.einsum("bhli,bhli->bhl", [mh_layer_output.grad, mh_layer_output])
        #         head_importance[layer] += dot.abs().sum(-1).sum(0).detach()
        #
        # # Also store our logits/labels if we want to compute metrics afterwards
        # if preds is None:
        #     preds = logits.detach().cpu().numpy()
        #     labels = label_ids.detach().cpu().numpy()
        # else:
        #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        #     labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)
        # #
        # tot_tokens += input_mask.float().detach().sum().data
        #
        # # Normalize
        # attn_entropy /= tot_tokens
        # head_importance /= tot_tokens
        # # Layerwise importance normalization
        # if not args.dont_normalize_importance_by_layer:
        #     exponent = 2
        #     norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1/exponent)
        #     head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
        #
        # if not args.dont_normalize_global_importance:
        #     head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())
    return att_probs, layer_entropies, attn_entropy, head_importance, preds, labels

def get_bert_att(Tokens, Indices, ds, mdl, fileid, words, save_matrices):
    start = 0
    Mat = np.zeros([len(words), len(words)])
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_file", default='bert_att.txt', type=str, required=False)
    parser.add_argument("--output_attentions", default=True, type=bool, required=False)
    parser.add_argument("--output_file", default='test_output_bert.txt', type=str, required=False)
    parser.add_argument("--bert_model", default=mdl, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    # if 'uncased' in mdl:
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--mode", default='client', type=str)
    parser.add_argument("--port", default=51940, type=int)
    parser.add_argument("--batch_size", default=250, type=int, help="Batch size for predictions.")

    parser.add_argument("--dont_normalize_importance_by_layer", action='store_true',
                        help="Don't normalize importance score by layers")
    parser.add_argument("--dont_normalize_global_importance", action='store_true',
                        help="Don't normalize all importance scores between 0 and 1")

    parser.add_argument("--try_masking", action='store_true',
                        help="Whether to try to mask head until a threshold of accuracy.")
    parser.add_argument("--masking_threshold", default=0.9, type=float, help="masking threshold in term of metrics"
                                                                             "(stop masking when metric < threshold * original metric value).")
    parser.add_argument("--masking_amount", default=0.1, type=float,
                        help="Amount to heads to masking at each masking step.")
    parser.add_argument("--metric_name", default="acc", type=str, help="Metric to use for head masking.")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup devices and distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(args.device, n_gpu, bool(args.local_rank != -1)))

    # Set seeds
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.seed)
    layer_indexes = [int(x) for x in args.layers.split(",")]
    if 'uncased' in mdl:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)

    model = BertModel.from_pretrained(args.bert_model, output_hidden_states=True, output_attentions=True)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only one distributed process download model & vocab
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)
    model.eval()
    for it, t in enumerate(Tokens):
        with open('bert_att.txt','w') as f:
            f.write(' '.join(t) +'\n')
        examples = read_examples(args.input_file)
        try:
            features = convert_examples_to_features(
                examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)
        except:
            return 0
        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature
        #model = BertForSequenceClassification.from_pretrained(args.bert_model,output_hidden_states=True, output_attentions=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
        print(args)
        model.eval()
        Idx = get_Idx(model,[t],[Indices[it]],all_input_ids,eval_dataloader,args.device,features,layer_indexes,mdl)
        att_probs, layer_entropies, attn_entropy, head_importance, _, _ = compute_heads_importance(args, model, eval_dataloader,features[0].tokens,Idx)

        Mat[start:start + len(att_probs[0]), start:start + len(att_probs[0])] = att_probs[0].numpy()
        start += len(att_probs[0])
    if save_matrices == True:
        path = 'vectors/' + ds + '/attention/' + mdl
        if not os.path.exists(path):
            os.makedirs(path)
        fileid = fileid.split('.')[0]
        fileid = fileid.split('-')[-1]
        writeMat(Mat.tolist(), path + '/Att-' + fileid)
    else:
        return Mat.tolist()