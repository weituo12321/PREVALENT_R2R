''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx


# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<NAV>', '<ORA>', '<TAR>']
padding_idx = base_vocab.index('<PAD>')

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits):
    data = []
    for split in splits:
        assert split in ['train', 'val_seen', 'val_unseen', 'test']
        with open('tasks/NDH/data/%s.json' % split) as f:
            data += json.load(f)
    return data


class BTokenizer(object):
    def __init__(self, encoding_length = 20, added_special_tokens=['<NAV>','ORA','TAR']):
        # <NAV>, <ORA>,<TAR>
        from pytorch_transformers import BertTokenizer
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        added_tok = {'additional_special_tokens': added_special_tokens}
        self.tokenizer.add_special_tokens(added_tok)
        self.encoding_length = encoding_length


    def encode_sentence(self, sentence, seps=None):
        txt = '[CLS] ' + sentence + ' [SEP]'
        encoding = self.tokenizer.encode(txt)
        if len(encoding) < self.encoding_length:
            encoding += [self.tokenizer.pad_token_id] * (self.encoding_length-len(encoding))

        # cut off the LHS of the encoding if it's over-size (e.g., words from the end of an individual command,
        # favoring those at the beginning of the command (since inst word order is reversed) (e.g., cut off the early
        # instructions in a dialog if the dialog is over size, preserving the latest QA pairs).
        prefix_cut = max(0, len(encoding) - self.encoding_length)
        return np.array(encoding[prefix_cut:])

    def decode_sentence(self, encoding):
        return self.tokenizer.decode(encoding)

    def __len__(self):
        return len(self.tokenizer)




class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentences, seps=None):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        if type(sentences) is not list:
            sentences = [sentences]
            seps = [seps]
        for sentence, sep in zip(sentences, seps):
            if sep is not None:
                encoding.append(self.word_to_index[sep])
            for word in self.split_sentence(sentence)[::-1]:  # reverse input sentences
                if word in self.word_to_index:
                    encoding.append(self.word_to_index[word])
                else:
                    encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))

        # cut off the LHS of the encoding if it's over-size (e.g., words from the end of an individual command,
        # favoring those at the beginning of the command (since inst word order is reversed) (e.g., cut off the early
        # instructions in a dialog if the dialog is over size, preserving the latest QA pairs).
        prefix_cut = max(0, len(encoding) - self.encoding_length)
        return np.array(encoding[prefix_cut:])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::-1]) # unreverse before output


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for turn in item['dialog_history']:
            count.update(t.split_sentence(turn['message']))
    vocab = list(start_vocab)

    # Add words that are object targets.
    targets = set()
    for item in data:
        target = item['target']
        targets.add(target)
    vocab.extend(list(targets))

    # Add words above min_count threshold.
    for word, num in count.most_common():
        if word in vocab:  # targets strings may also appear as regular vocabulary.
            continue
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def preprocess_get_pano_states(navigable_locs_path = "tasks/NDH/data/navigable_locs.json"):
    if os.path.exists(navigable_locs_path):
        return
    image_w = 640
    image_h = 480
    vfov = 60
    import sys
    sys.path.append('build')
    import MatterSim
    from collections import defaultdict

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(image_w, image_h)
    sim.setCameraVFOV(math.radians(vfov))
    sim.init()

    splits = ['train', 'val_seen', 'val_unseen', 'test']
    graphs = {}
    for split in splits:
        #data = load_datasets([split], encoder_type='lstm')
        data = load_datasets([split])
        for item in data:
            # print(item.keys())
            # print("")
            scan = item["scan"]
            if scan in graphs:
                continue
            graphs[scan] = {}
            with open('connectivity/%s_connectivity.json' % scan) as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    if item['included']:
                        viewpointId = item['image_id']
                        sim.newEpisode(scan, viewpointId, 0, 0)
                        state = sim.getState()

                        initViewIndex = state.viewIndex
                        # 1. first look down, turning to relViewIndex 0
                        elevation_delta = -(state.viewIndex // 12)
                        for _ in range(int(abs(elevation_delta))):
                            ''' Make possibly more than one elevation turns '''
                            sim.makeAction(0, 0, np.sign(elevation_delta))

                        adj_dict = {}
                        for relViewIndex in range(36):
                            state = sim.getState()
                            absViewIndex = state.viewIndex
                            for loc in state.navigableLocations[1:]:
                                distance = _loc_distance(loc)
                                if (loc.viewpointId not in adj_dict or
                                    distance < adj_dict[loc.viewpointId]['distance']):
                                    adj_dict[loc.viewpointId] = {
                                        'absViewIndex': absViewIndex,
                                        'nextViewpointId': loc.viewpointId,
                                        'loc_rel_heading': loc.rel_heading,
                                        'loc_rel_elevation': loc.rel_elevation,
                                        'distance': distance}
                            if (relViewIndex + 1) % 12 == 0:
                                sim.makeAction(0, 1, 1)  # Turn right and look up
                            else:
                                sim.makeAction(0, 1, 0)  # Turn right
                        # 3. turn back to the original view
                        for _ in range(int(abs(- 2 - elevation_delta))):
                            ''' Make possibly more than one elevation turns '''
                            sim.makeAction(0, 0, np.sign(- 2 - elevation_delta))

                        state = sim.getState()
                        assert state.viewIndex == initViewIndex

                        absViewIndex2points = defaultdict(list)
                        for vpId, point in adj_dict.items():
                            absViewIndex2points[point['absViewIndex']].append(vpId)
                        graphs[scan][viewpointId]=(adj_dict, absViewIndex2points)
        print('prepare cache for', split, 'done')
    with open(navigable_locs_path, 'w') as f:
        json.dump(graphs, f)

def _loc_distance(loc):
    return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)


def current_best(df, v_id, best_score_name):
    if best_score_name == 'sr_sum':
        return  df['val_seen success_rate'][v_id] + df['val_unseen success_rate'][v_id]
    elif best_score_name == 'spl_sum':
        return  df['val_seen spl'][v_id] + df['val_unseen spl'][v_id]
    elif best_score_name == 'spl_unseen':
        return  df['val_unseen spl'][v_id]
    elif best_score_name == 'sr_unseen':
        return  df['val_unseen success_rate'][v_id]
    elif best_score_name == 'dr_unseen':
        return  df['val_unseen dist_to_end_reduction'][v_id]
