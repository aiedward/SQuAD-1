from pycorenlp import StanfordCoreNLP

## Connect to CoreNLP server
print "Connecting to CoreNLP server..."    
nlp = StanfordCoreNLP('http://localhost:9000')
print "Connected!"    
print(len(all_context_lines))

output.append(nlp.annotate(words_list, properties={
    'annotators': 'pos,ner,lemma',
    'tokenize.language': 'Whitespace',
    'outputFormat': 'json'}))
print('Finished annotation!')

pos2int = {"CC":0, "CD":1, "DT":2, "EX":3, "FW":4, "IN":5, "JJ":6, "JJR":7, "JJS":8, \
    "LS":9, "MD":10, "NN":11, "NNS":12, "NNP":13, "NNPS":14, "PDT":15, "POS":16, \
    "PRP":17, "PRP$":18, "RB":19, "RBR":20, "RBS":21, "RP":22, "SYM":23, "TO":24, \
    "UH":25, "VB":26, "VBD":27, "VBG":28, "VBN":29, "VBP":30, "VBZ":31, "WDT":32, \
    "WP":33, "WP$":34, "WRB":35}

ner2int = {"O":0, "PERSON":1, "LOCATION":2, "ORGANIZATION":3, "MISC":4, "MONEY":5, \
           "NUMBER":6, "ORDINAL":7, "PERCENT":8, "DATE":9, "TIME":10, "DURATION":11, "SET":12, \
           "EMAIL":13, "URL":14, "CITY":15, "STATE_OR_PROVINCE":16, "COUNTRY":17, "NATIONALITY":18, \
           "RELIGION":19, "TITLE":20, "IDEOLOGY":21, "CRIMINAL_CHARGE":22, "CAUSE_OF_DEATH":23}

import pickle
pickle.dump( output, open( "output.p", "wb" ) )

ner_tags = np.array([ner2int[str(tok['ner'])] if str(tok['ner']) in ner2int.keys() else 0 for s in output['sentences'] for tok in s['tokens']])  # length = total number of tokens in batch
pos_tags = np.array([pos2int[str(tok['pos'])] if str(tok['pos']) in pos2int.keys() else -1 for s in output['sentences'] for tok in s['tokens']]) # length = total number of tokens in batch
lems     = [tok['lemma'] for s in output['sentences'] for tok in s['tokens']]

assert (len(ner_tags)==total_tokens), "%d, %d" % (len(ner_tags), total_tokens)
assert (len(pos_tags)==total_tokens), "%d, %d" % (len(pos_tags), total_tokens)
assert (len(lems)==total_tokens),     "%d, %d" % (len(lems),     total_tokens)

examples_feats = []
for ex in examples:
    context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens = ex
    num_context_tokens = len(context_tokens)
    ner_tags_ex = ner_tags[:num_context_tokens]
    pos_tags_ex = pos_tags[:num_context_tokens]
    lems_ex     = lems[:num_context_tokens]

    del ner_tags[:num_context_tokens]
    del pos_tags[:num_context_tokens]
    del lems[:num_context_tokens]

    # compare each context token to query tokens
    match_orig  = np.array([int(any(context_token==q         for q in qn_tokens)) for context_token in context_tokens]) # original form
    match_lower = np.array([int(any(context_token.lower()==q for q in qn_tokens)) for context_token in context_tokens]) # lower case
    match_lemma = np.array([int(any(context_token_lem==q     for q in qn_tokens)) for context_token_lem in lems_ex])    # lemma form
    feats = np.concatenate([np.expand_dims(x,1) for x in (pos_tags_ex, ner_tags_ex, match_orig, match_lower, match_lemma)], axis=1)  # (N,5)
    examples_feats.append((context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens, feats))
examples = examples_feats





######## Concatenate all (non-discarded) lines from this batch so we only have to do one annotation ########
all_context_lines += this_context_line.rstrip()
all_context_lines += " "