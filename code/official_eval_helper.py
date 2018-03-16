# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This code is required for "official_eval" mode in main.py
It provides functions to read a SQuAD json file, use the model to get predicted answers,
and write those answers to another JSON file."""

from __future__ import absolute_import
from __future__ import division

import os
from tqdm import tqdm
import numpy as np
from six.moves import xrange
from nltk.tokenize.moses import MosesDetokenizer

from preprocessing.squad_preprocess import data_from_json, tokenize
from vocab import UNK_ID, PAD_ID
from data_batcher import padded, Batch, padded2, get_wordnet_pos

from nltk import pos_tag, ne_chunk, FreqDist
from nltk.chunk import tree2conlltags
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def readnext(x):
    """x is a list"""
    if len(x) == 0:
        return False
    else:
        return x.pop(0)

def refill_batches(batches, word2id, qn_uuid_data, context_token_data, qn_token_data, batch_size, context_len, question_len, word_len, mcids_dict):
    """
    This is similar to refill_batches in data_batcher.py, but:
      (1) instead of reading from (preprocessed) datafiles, it reads from the provided lists
      (2) it only puts the context and question information in the batches (not the answer information)
      (3) it also gets UUID information and puts it in the batches

    Inputs:
      batches: list to be refilled
      qn_uuid_data: list of strings that are unique ids
      context_token_data, qn_token_data: list of lists of strings (no UNKs, no padding)
      batch_size: int. size of batches to make
      context_len, question_len: ints. max sizes of context and question. Anything longer is truncated.

    Makes batches that contain:
      uuids_batch, context_tokens_batch, context_ids_batch, qn_ids_batch: all lists length batch_size
    """
    examples = []

    # Get next example
    qn_uuid, context_tokens, qn_tokens = readnext(qn_uuid_data), readnext(context_token_data), readnext(qn_token_data)

    pos2int = {"CC":0, "CD":1, "DT":2, "EX":3, "FW":4, "IN":5, "JJ":6, "JJR":7, "JJS":8, \
        "LS":9, "MD":10, "NN":11, "NNS":12, "NNP":13, "NNPS":14, "PDT":15, "POS":16, \
        "PRP":17, "PRP$":18, "RB":19, "RBR":20, "RBS":21, "RP":22, "SYM":23, "TO":24, \
        "UH":25, "VB":26, "VBD":27, "VBG":28, "VBN":29, "VBP":30, "VBZ":31, "WDT":32, \
        "WP":33, "WP$":34, "WRB":35}
    ner2int = {"O":0, "PERSON":1, "LOCATION":2, "ORGANIZATION":3, "GSP":4, "GPE":5, "FACILITY":6}
    pos_keys = pos2int.keys()
    ner_keys = ner2int.keys() 
    lemmatizer = WordNetLemmatizer()
    a = 0.4

    char2id = {"a":2, "b":3, "c":4, "d":5, "e":6, "f":7, "g":8, \
        "h":9, "i":10, "j":11, "k":12, "l":13, "m":14, "n":15, "o":16, \
        "p":17, "q":18, "r":19, "s":20, "t":21, "u":22, "v":23, "w":24, \
        "x":25, "y":26, "z":27, "0":28, "1":29, "2":30, "3":31, "4":32, \
        "5":33, "6":34, "7":35, "8":36, "9":37, ".":38, ",":39, '"':40, \
        "?":41, "'":42}
    char_keys = char2id.keys()

    mcids_keys = mcids_dict.keys()

    while qn_uuid and context_tokens and qn_tokens:

        ########## GENERATE CHARACTER TOKENS #########################
        char_ids = [[char2id[char] if char in char_keys else UNK_ID for char in tok.lower()] for tok  in context_tokens]
        char_ids = [x[:word_len] for x in char_ids] # (N, <=word_len)
        char_ids = padded(char_ids, word_len) # (N, word_len)

        charQ_ids = [[char2id[char] if char in char_keys else UNK_ID for char in tok.lower()] for tok  in qn_tokens]
        charQ_ids = [x[:word_len] for x in charQ_ids] # (M, <=word_len)
        charQ_ids = padded(charQ_ids, word_len) # (M, word_len)
        ##############################################################

        ########## GET COMMONQ EMBEDDING INDICES AND MASK ############
        commonQ_mask        = [x in mcids_keys for x in qn_ids] # (M)
        commonQ_emb_indices = [mcids_dict.get(x,0) for x in qn_ids] # (M)

        commonC_mask        = [x in mcids_keys for x in context_ids] # (N)
        commonC_emb_indices = [mcids_dict.get(x,0) for x in context_ids] # (N) - note the 0 index doesnt matter due to mask
        ##############################################################

        ########## GENERATE EXACT MATCH + POS/NER FEATURES ###########
        # calculate POS and NER tags (as strings)
        pos_tree = pos_tag(context_tokens)
        pos_tags = [p[1] for p in pos_tree]
        # chunk = ne_chunk(pos_tree)
        # ner_tags = [ne[2][2:] for ne in tree2conlltags(chunk)]

        # convert POS and NER tags to ints using dictionary
        pos_ids = [pos2int[pos] if pos in pos_keys else -1 for pos in pos_tags]
        # ner_ids = [ner2int[ne]  if ne  in ner_keys else 0  for ne  in ner_tags]

        # compute lemmatized version of each context token                
        lems = [str(lemmatizer.lemmatize(tok,get_wordnet_pos(pos))) if get_wordnet_pos(pos) else str(lemmatizer.lemmatize(tok)) for tok,pos in zip(context_tokens,pos_tags)]

        # compare each context word to query words for three different versions
        match_orig  = [int(sum([context_token==q     for q in qn_tokens])==1) for context_token     in context_tokens] # original form
        match_lemma = [int(sum([context_token_lem==q for q in qn_tokens])==1) for context_token_lem in lems]    # lemma form

        # compute normalized term frequency
        fdist = FreqDist(context_tokens)
        max_count = float(max(fdist.values()))
        tf = [a + (1-a)*fdist[w]/max_count for w in context_tokens]

        # feats = zip(*(pos_ids, ner_ids, match_orig, match_lemma))  # (N,4)
        # feats = zip(*(pos_ids, match_orig, match_lemma))  # (N,3)
        feats = zip(*(pos_ids, tf, match_orig, match_lemma))  # (N,4)
        ##############################################################

        # Convert context_tokens and qn_tokens to context_ids and qn_ids
        context_ids = [word2id.get(w, UNK_ID) for w in context_tokens]
        qn_ids = [word2id.get(w, UNK_ID) for w in qn_tokens]

        # Truncate context_ids and qn_ids
        # Note: truncating context_ids may truncate the correct answer, meaning that it's impossible for your model to get the correct answer on this example!
        if len(qn_ids) > question_len:
            qn_ids = qn_ids[:question_len]
            commonQ_mask = commonQ_mask[:question_len]
            commonQ_emb_indices = commonQ_emb_indices[:question_len]
            charQ_ids = charQ_ids[:question_len]
        if len(context_ids) > context_len:
            context_ids = context_ids[:context_len]
            feats = feats[:context_len]
            char_ids = char_ids[:context_len]
            commonC_mask = commonC_mask[:context_len]
            commonC_emb_indices = commonC_emb_indices[:context_len]

        # Add to list of examples
        examples.append((qn_uuid, context_tokens, context_ids, qn_ids, feats, char_ids, commonQ_mask, commonQ_emb_indices, charQ_ids, commonC_mask, commonC_emb_indices))

        # Stop if you've got a batch
        if len(examples) == batch_size:
            break

        # Get next example
        qn_uuid, context_tokens, qn_tokens = readnext(qn_uuid_data), readnext(context_token_data), readnext(qn_token_data)

    # Make into batches
    for batch_start in xrange(0, len(examples), batch_size):
        uuids_batch, context_tokens_batch, context_ids_batch, qn_ids_batch, feats_batch, char_ids_batch, commonQ_mask_batch, commonQ_emb_indices_batch, charQ_ids_batch, commonC_mask_batch, commonC_emb_indices_batch = zip(*examples[batch_start:batch_start + batch_size])

        batches.append((uuids_batch, context_tokens_batch, context_ids_batch, qn_ids_batch, feats_batch, char_ids_batch, commonQ_mask_batch, commonQ_emb_indices_batch, charQ_ids_batch, commonC_mask_batch, commonC_emb_indices_batch))

    return



def get_batch_generator(word2id, qn_uuid_data, context_token_data, qn_token_data, batch_size, context_len, question_len, num_feats, word_len, mcids_dict):
    """
    This is similar to get_batch_generator in data_batcher.py, but with some
    differences (see explanation in refill_batches).

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      qn_uuid_data: list of strings that are unique ids
      context_token_data, qn_token_data: list of lists of strings (no UNKs, no padding)
      batch_size: int. size of batches to make
      context_len, question_len: ints. max sizes of context and question. Anything longer is truncated.

    Yields:
      Batch objects, but they only contain context and question information (no answer information)
    """
    batches = []

    while True:
        if len(batches) == 0:
            refill_batches(batches, word2id, qn_uuid_data, context_token_data, qn_token_data, batch_size, context_len, question_len, word_len, mcids_dict)
        if len(batches) == 0:
            break

        # Get next batch. These are all lists length batch_size
        (uuids, context_tokens, context_ids, qn_ids, feats, char_ids, commonQ_mask, commonQ_emb_indices, commonC_mask, commonC_emb_indices) = batches.pop(0)

        # Pad context_ids and qn_ids
        qn_ids = padded(qn_ids, question_len) # pad questions to length question_len
        context_ids = padded(context_ids, context_len) # pad contexts to length context_len

        # Make qn_ids into a np array and create qn_mask
        qn_ids = np.array(qn_ids)
        qn_mask = (qn_ids != PAD_ID).astype(np.int32)

        # Make context_ids into a np array and create context_mask
        context_ids = np.array(context_ids)
        context_mask = (context_ids != PAD_ID).astype(np.int32)

        # Make feats into an np array
        feats = np.array(padded2(feats, num_feats, context_len))

        # Pad character ids (first for word length, then for context length), then make into array
        char_ids = padded2(char_ids, word_len, context_len, islist=True)
        char_ids = np.array(char_ids)
        char_mask = (char_ids != PAD_ID).astype(np.int32)

        charQ_ids = padded2(charQ_ids, word_len, question_len, islist=True)
        charQ_ids = np.array(charQ_ids)
        charQ_mask = (charQ_ids != PAD_ID).astype(np.int32)

        # Pad commonQ_mask and commonQ_emb_indices / convert to np.array
        commonQ_mask = np.array(paddedBool(commonQ_mask, question_len))
        commonQ_emb_indices = np.array(padded(commonQ_emb_indices, question_len))

        commonC_mask = np.array(paddedBool(commonC_mask, context_len))
        commonC_emb_indices = np.array(padded(commonC_emb_indices, context_len))

        # Make into a Batch object
        batch = Batch(context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens=None, ans_span=None, ans_tokens=None, \
            feats=feats, char_ids=char_ids, char_mask=char_mask, commonQ_mask=commonQ_mask, commonQ_emb_indices=commonQ_emb_indices, \
            charQ_ids=charQ_ids, charQ_mask=charQ_mask, commonC_mask=commonC_mask, commonC_emb_indices=commonC_emb_indices, uuids=uuids)

        yield batch

    return


def preprocess_dataset(dataset):
    """
    Note: this is similar to squad_preprocess.preprocess_and_write, but:
      (1) We only extract the context and question information from the JSON file.
        We don't extract answer information. This makes this function much simpler
        than squad_preprocess.preprocess_and_write, because we don't have to convert
        the character spans to word spans. This also means that we don't have to
        discard any examples due to tokenization problems.

    Input:
      dataset: data read from SQuAD JSON file

    Returns:
      qn_uuid_data, context_token_data, qn_token_data: lists of uuids, tokenized context and tokenized questions
    """
    qn_uuid_data = []
    context_token_data = []
    qn_token_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing data"):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):

            context = unicode(article_paragraphs[pid]['context']) # string

            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context) # list of strings (lowercase)
            context = context.lower()

            qas = article_paragraphs[pid]['qas'] # list of questions

            # for each question
            for qn in qas:

                # read the question text and tokenize
                question = unicode(qn['question']) # string
                question_tokens = tokenize(question) # list of strings

                # also get the question_uuid
                question_uuid = qn['id']

                # Append to data lists
                qn_uuid_data.append(question_uuid)
                context_token_data.append(context_tokens)
                qn_token_data.append(question_tokens)

    return qn_uuid_data, context_token_data, qn_token_data


def get_json_data(data_filename):
    """
    Read the contexts and questions from a .json file (like dev-v1.1.json)

    Returns:
      qn_uuid_data: list (length equal to dev set size) of unicode strings like '56be4db0acb8001400a502ec'
      context_token_data, qn_token_data: lists (length equal to dev set size) of lists of strings (no UNKs, unpadded)
    """
    # Check the data file exists
    if not os.path.exists(data_filename):
        raise Exception("JSON input file does not exist: %s" % data_filename)

    # Read the json file
    print "Reading data from %s..." % data_filename
    data = data_from_json(data_filename)

    # Get the tokenized contexts and questions, and unique question identifiers
    print "Preprocessing data from %s..." % data_filename
    qn_uuid_data, context_token_data, qn_token_data = preprocess_dataset(data)

    data_size = len(qn_uuid_data)
    assert len(context_token_data) == data_size
    assert len(qn_token_data) == data_size
    print "Finished preprocessing. Got %i examples from %s" % (data_size, data_filename)

    return qn_uuid_data, context_token_data, qn_token_data


def generate_answers(session, model, word2id, qn_uuid_data, context_token_data, qn_token_data):
    """
    Given a model, and a set of (context, question) pairs, each with a unique ID,
    use the model to generate an answer for each pair, and return a dictionary mapping
    each unique ID to the generated answer.

    Inputs:
      session: TensorFlow session
      model: QAModel
      word2id: dictionary mapping word (string) to word id (int)
      qn_uuid_data, context_token_data, qn_token_data: lists

    Outputs:
      uuid2ans: dictionary mapping uuid (string) to predicted answer (string; detokenized)
    """
    uuid2ans = {} # maps uuid to string containing predicted answer
    data_size = len(qn_uuid_data)
    num_batches = ((data_size-1) / model.FLAGS.batch_size) + 1
    batch_num = 0
    detokenizer = MosesDetokenizer()

    print "Generating answers..."

    for batch in get_batch_generator(word2id, qn_uuid_data, context_token_data, qn_token_data, model.FLAGS.batch_size, model.FLAGS.context_len, model.FLAGS.question_len, model.FLAGS.num_feats, model.FLAGS.word_len, model.mcids_dict):

        # Get the predicted spans
        pred_start_batch, pred_end_batch = model.get_start_end_pos(session, batch, model.FLAGS.max_span)

        # Convert pred_start_batch and pred_end_batch to lists length batch_size
        pred_start_batch = pred_start_batch.tolist()
        pred_end_batch = pred_end_batch.tolist()

        # For each example in the batch:
        for ex_idx, (pred_start, pred_end) in enumerate(zip(pred_start_batch, pred_end_batch)):

            # Original context tokens (no UNKs or padding) for this example
            context_tokens = batch.context_tokens[ex_idx] # list of strings

            # Check the predicted span is in range
            assert pred_start in range(len(context_tokens))
            assert pred_end in range(len(context_tokens))

            # Predicted answer tokens
            pred_ans_tokens = context_tokens[pred_start : pred_end +1] # list of strings

            # Detokenize and add to dict
            uuid = batch.uuids[ex_idx]
            uuid2ans[uuid] = detokenizer.detokenize(pred_ans_tokens, return_str=True)

        batch_num += 1

        if batch_num % 10 == 0:
            print "Generated answers for %i/%i batches = %.2f%%" % (batch_num, num_batches, batch_num*100.0/num_batches)

    print "Finished generating answers for dataset."

    return uuid2ans
