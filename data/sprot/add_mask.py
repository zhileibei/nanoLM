import pickle
meta_path = "/home/beizl42/orcd/pool/datasets/uniprotkb/meta.pkl"
meta = pickle.load(open(meta_path, 'rb'))
meta['vocab_size'] += 1  # for mask token
meta['itos'][24] = '<MASK>'
meta['stoi']['<MASK>'] = 24
breakpoint()
pickle.dump(meta, open(meta_path, 'wb'))