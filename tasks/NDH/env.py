''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx
import itertools
import pdb

from utils import load_datasets, load_nav_graphs

csv.field_size_limit(sys.maxsize)

"""

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100, blind=False):
        if feature_store:
            print 'Loading image features from %s' % feature_store
            if blind:
                print("... and zeroing them out for 'blind' evaluation")
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
            self.features = {}
            with open(feature_store, "r+b") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
                for item in reader:
                    self.image_h = int(item['image_h'])
                    self.image_w = int(item['image_w'])
                    self.vfov = int(item['vfov'])
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    if not blind:
                        self.features[long_id] = np.frombuffer(base64.decodestring(item['features']),
                                dtype=np.float32).reshape((36, 2048))
                    else:
                        self.features[long_id] = np.zeros((36, 2048), dtype=np.float32)
        else:
            print 'Image features not provided'
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.batch_size = batch_size
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setBatchSize(self.batch_size)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.sim.initialize()

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        self.sim.newEpisode(scanIds, viewpointIds, headings, [0]*self.batch_size)

    def getStates(self):
        ''' Get list of states augmented with precomputed image features. rgb field will be empty. '''
        feature_states = []
        for state in self.sim.getState():
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id][state.viewIndex,:]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        ix = []
        heading = []
        elevation = []
        for i,h,e in actions:
            ix.append(int(i))
            heading.append(float(h))
            elevation.append(float(e))
        self.sim.makeAction(ix, heading, elevation)

    def makeSimpleActions(self, simple_indices):
        ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. '''
        actions = []
        for i, index in enumerate(simple_indices):
            if index == 0:
                actions.append((1, 0, 0))
            elif index == 1:
                actions.append((0,-1, 0))
            elif index == 2:
                actions.append((0, 1, 0))
            elif index == 3:
                actions.append((0, 0, 1))
            elif index == 4:
                actions.append((0, 0,-1))
            else:
                sys.exit("Invalid simple action");
        self.makeActions(actions)
"""

debug_beam = False  # if True, even beam_size=1, still use beam_search
class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100, beam_size=1, blind=False):
        self.feature = feature_store
        self.image_w = self.feature.image_w
        self.image_h = self.feature.image_h
        self.vfov = self.feature.vfov
        self.blind = blind
        # if feature_store:
        #     print('Loading image features from %s' % feature_store)
        #     tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
        #     self.features = {}
        #     with open(feature_store, "r+") as tsv_in_file:
        #         reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = tsv_fieldnames)
        #         for item in reader:
        #             self.image_h = int(item['image_h'])
        #             self.image_w = int(item['image_w'])
        #             self.vfov = int(item['vfov'])
        #             long_id = self._make_id(item['scanId'], item['viewpointId'])
        #             self.features[long_id] = np.frombuffer(base64.b64decode(item['features']),
        #                     dtype=np.float32).reshape((36, 2048))
        # else:
        #     print('Image features not provided')
        #     self.features = self.feature

        self.sims = []
        for i in range(batch_size):
            if beam_size == 1 and not debug_beam:
                sim = MatterSim.Simulator()
                sim.setRenderingEnabled(False)
                sim.setDiscretizedViewingAngles(True)
                sim.setCameraResolution(self.image_w, self.image_h)
                sim.setCameraVFOV(math.radians(self.vfov))
                sim.init()
                self.sims.append(sim)
            else:
                sims = []
                for ii in range(beam_size):
                    sim = MatterSim.Simulator()
                    sim.setRenderingEnabled(False)
                    sim.setDiscretizedViewingAngles(True)
                    sim.setCameraResolution(self.image_w, self.image_h)
                    sim.setCameraVFOV(math.radians(self.vfov))
                    sim.init()
                    sims.append(sim)
                self.sims.append(sims)

        #self.sim_dict = {}

    def sims_view(self):
        return [itertools.cycle(sim_list) for sim_list in self.sims]

    # def _make_id(self, scanId, viewpointId):  # jolin
    #     return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        feature_states = []
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            if isinstance(self.sims[i], list):  # beamed
                world_state = WorldState(scanId, viewpointId, heading, 0)
                self.sims[i][0].newEpisode(*world_state)
                # self.sims[i][0].newEpisode(scanId, viewpointId, heading, 0)
                state = self.sims[i][0].getState()
                feature = self.feature.rollout(state.scanId, state.location.viewpointId, state.viewIndex)
                feature_states.append([(feature, world_state)])
            else:
                self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
                state = self.sims[i].getState()
                feature = self.feature.rollout(state.scanId, state.location.viewpointId, state.viewIndex)
                if self.blind: # debug?
                    features = np.zeros((36, 2048), dtype=np.float32)
                feature_states.append((feature, state, self.sims[i]))
        return feature_states

    def newBatchEpisodes(self, svhs):
        feature_states = []
        for i, svh in enumerate(svhs):
            scanId = svh[0]
            viewpointId = svh[1]
            heading = svh[2]

            if isinstance(self.sims[i], list):  # beamed
                world_state = WorldState(scanId, viewpointId, heading, 0)
                self.sims[i][0].newEpisode(*world_state)
                # self.sims[i][0].newEpisode(scanId, viewpointId, heading, 0)
                state = self.sims[i][0].getState()
                feature = self.feature.rollout(state.scanId, state.location.viewpointId, state.viewIndex)
                feature_states.append([(feature, world_state)])
            else:
                self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
                state = self.sims[i].getState()
                feature = self.feature.rollout(state.scanId, state.location.viewpointId, state.viewIndex)
                if self.blind: # debug?
                    features = np.zeros((36, 2048), dtype=np.float32)

                feature_states.append((feature, state, self.sims[i]))
        return feature_states



    def pre_loadSims(self, data):
        """ debug: pre-load all the sims """

        for traj in data:
            scanId = traj['scan']
            vpId = traj['path'][0]
            heading = traj['heading']

            path_id = traj['path_id']

            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()

            sim.newEpisode(scanId, vpId, heading, 0)
            state = sim.getState()
            feature = self.feature.rollout(state.scanId, state.location.viewpointId, state.viewIndex)
            self.sim_dict[path_id] = (feature, state, sim)

        print("load sims done: %d" % (len(data)))

    def batchEpisodes(self, pathIds, scanIds, viewpointIds, headings):
        """ debug: sample a batch of sims """

        feature_states = []
        self.sims.clear()

        for i, (pathId, scanId, viewpointId, heading) in enumerate(zip(pathIds, scanIds, viewpointIds, headings)):
            if pathId in self.sim_dict: # debug
                feature_states.append(self.sim_dict[pathId])
                self.sims.append(copy.deepcopy(self.sim_dict[pathId][2]))

        return feature_states



    def getStates(self):
        ''' Get list of states augmented with precomputed image features. rgb field will be empty. '''
        feature_states = []
        for sim in self.sims:
            if isinstance(sim, list):  # beamed
                feature_states_inside = []
                for si in sim:
                    state = si.getState()
                    # jolin
                    feature = self.feature.rollout(state.scanId,
                                                   state.location.viewpointId,
                                                   state.viewIndex)
                    feature_states_inside.append((feature, state, si))
                feature_states.append(feature_states_inside)
            else:
                state = sim.getState()
                # jolin
                feature = self.feature.rollout(state.scanId,
                                               state.location.viewpointId,
                                               state.viewIndex)
                feature_states.append((feature, state, sim))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

    def makeSimpleActions(self, simple_indices):
        ''' Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. '''
        for i, index in enumerate(simple_indices):
            if index == 0:
                self.sims[i].makeAction(1, 0, 0)
            elif index == 1:
                self.sims[i].makeAction(0,-1, 0)
            elif index == 2:
                self.sims[i].makeAction(0, 1, 0)
            elif index == 3:
                self.sims[i].makeAction(0, 0, 1)
            elif index == 4:
                self.sims[i].makeAction(0, 0,-1)
            else:
                sys.exit("Invalid simple action")


angle_inc = np.pi / 6.


def build_viewpoint_loc_embedding(viewIndex):
    """
    Position embedding:
    heading 64D + elevation 64D
    1) heading: [sin(heading) for _ in range(1, 33)] +
                [cos(heading) for _ in range(1, 33)]
    2) elevation: [sin(elevation) for _ in range(1, 33)] +
                  [cos(elevation) for _ in range(1, 33)]
    """
    embedding = np.zeros((36, 128), np.float32)
    for absViewIndex in range(36):
        relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
        rel_heading = (relViewIndex % 12) * angle_inc
        rel_elevation = (relViewIndex // 12 - 1) * angle_inc
        embedding[absViewIndex,  0:32] = np.sin(rel_heading)
        embedding[absViewIndex, 32:64] = np.cos(rel_heading)
        embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
        embedding[absViewIndex,   96:] = np.cos(rel_elevation)
    return embedding


# pre-compute all the 36 possible paranoram location embeddings
_static_loc_embeddings = [
    build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]


def _loc_distance(loc):
    return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)


def _canonical_angle(x):
    ''' Make angle in (-pi, +pi) '''
    return x - 2 * np.pi * round(x / (2 * np.pi))


def _adjust_heading(sim, heading):
    heading = (heading + 6) % 12 - 6  # minimum action to turn (e.g 11 -> -1)
    ''' Make possibly more than one heading turns '''
    for _ in range(int(abs(heading))):
        sim.makeAction(0, np.sign(heading), 0)


def _adjust_elevation(sim, elevation):
    for _ in range(int(abs(elevation))):
        ''' Make possibly more than one elevation turns '''
        sim.makeAction(0, 0, np.sign(elevation))


def _navigate_to_location(sim, nextViewpointId, absViewIndex):
    state = sim.getState()
    if state.location.viewpointId == nextViewpointId:
        return  # do nothing

    # 1. Turn to the corresponding view orientation
    _adjust_heading(sim, absViewIndex % 12 - state.viewIndex % 12)
    _adjust_elevation(sim, absViewIndex // 12 - state.viewIndex // 12)
    # find the next location
    state = sim.getState()
    assert state.viewIndex == absViewIndex
    a, next_loc = None, None
    for n_loc, loc in enumerate(state.navigableLocations):
        if loc.viewpointId == nextViewpointId:
            a = n_loc
            next_loc = loc
            break
    assert next_loc is not None

    # 3. Take action
    sim.makeAction(a, 0, 0)


def _get_panorama_states(state, sim, nav_graphs):
    '''
    Look around and collect all the navigable locations

    Representation of all_adj_locs:
        {'absViewIndex': int,
         'relViewIndex': int,
         'nextViewpointId': int,
         'rel_heading': float,
         'rel_elevation': float}
        where relViewIndex is normalized using the current heading

    Concepts:
        - absViewIndex: the absolute viewpoint index, as returned by
          state.viewIndex
        - nextViewpointId: the viewpointID of this adjacent point
        - rel_heading: the heading (radians) of this adjacent point
          relative to looking forward horizontally (i.e. relViewIndex 12)
        - rel_elevation: the elevation (radians) of this adjacent point
          relative to looking forward horizontally (i.e. relViewIndex 12)

    Features are 36 x D_vis, ordered from relViewIndex 0 to 35 (i.e.
    feature[12] is always the feature of the patch forward horizontally)
    '''
    initViewIndex = state.viewIndex
    offset = initViewIndex%12
    absViewIndexDict = [i for i in range(offset, 12)] + [i for i in range(offset)]
    absViewIndexDict = [level*12 + i for level in range(3) for i in absViewIndexDict]
    try:
        adj_dict_cache, absViewIndex2points = nav_graphs[state.scanId][state.location.viewpointId]
        for relViewIndex in range(36):
            absViewIndex = absViewIndexDict[relViewIndex]
            if str(absViewIndex) not in absViewIndex2points:
                continue
            base_rel_heading = (relViewIndex % 12) * angle_inc
            base_rel_elevation = (relViewIndex // 12 - 1) * angle_inc
            adj_viewpointIds = absViewIndex2points[str(absViewIndex)]
            for point in adj_viewpointIds:
                adj_dict_cache[point]['rel_heading'] = _canonical_angle(base_rel_heading + adj_dict_cache[point]['loc_rel_heading'])
                adj_dict_cache[point]['rel_elevation'] = adj_dict_cache[point]['loc_rel_elevation'] + base_rel_elevation
        adj_dict = adj_dict_cache
    except KeyError:
        print("This should not happen")# should not happen
        raise
        # 1. first look down, turning to relViewIndex 0
        elevation_delta = -(state.viewIndex // 12)
        for _ in range(int(abs(elevation_delta))):
            ''' Make possibly more than one elevation turns '''
            sim.makeAction(0, 0, np.sign(elevation_delta))

        # 2. scan through the 36 views and collect all navigable locations
        adj_dict = {}
        for relViewIndex in range(36):
            # Here, base_rel_heading and base_rel_elevation are w.r.t
            # relViewIndex 12 (looking forward horizontally)
            # (i.e. the relative heading and elevation
            # adjustment needed to switch from relViewIndex 12
            # to the current relViewIndex)
            base_rel_heading = (relViewIndex % 12) * angle_inc
            base_rel_elevation = (relViewIndex // 12 - 1) * angle_inc

            state = sim.getState()
            absViewIndex = state.viewIndex
            # get adjacent locations
            for loc in state.navigableLocations[1:]:
                distance = _loc_distance(loc)
                # if a loc is visible from multiple view, use the closest
                # view (in angular distance) as its representation
                if (loc.viewpointId not in adj_dict or
                        distance < adj_dict[loc.viewpointId]['distance']):
                    rel_heading = _canonical_angle(
                        base_rel_heading + loc.rel_heading)
                    rel_elevation = base_rel_elevation + loc.rel_elevation
                    adj_dict[loc.viewpointId] = {
                        'absViewIndex': absViewIndex,
                        'nextViewpointId': loc.viewpointId,
                        'rel_heading': rel_heading,
                        'rel_elevation': rel_elevation,
                        'distance': distance}
            # move to the next view
            if (relViewIndex + 1) % 12 == 0:
                sim.makeAction(0, 1, 1)  # Turn right and look up
            else:
                sim.makeAction(0, 1, 0)  # Turn right
        # 3. turn back to the original view
        for _ in range(int(abs(- 2 - elevation_delta))):
            ''' Make possibly more than one elevation turns '''
            sim.makeAction(0, 0, np.sign(- 2 - elevation_delta))

        state = sim.getState()
        assert state.viewIndex == initViewIndex  # check the agent is back
    # collect navigable location list
    stop = {'absViewIndex': -1, 'nextViewpointId': state.location.viewpointId}
    # for viewpointId, point in adj_dict.items():
    #     for key in point:
    #         assert (point[key]==adj_dict_cache[viewpointId][key])
    adj_loc_list = [stop] + sorted(adj_dict.values(), key=lambda x: abs(x['rel_heading']))

    return adj_loc_list


def _build_action_embedding(adj_loc_list, features, skip_loc = 0):
    feature_dim = features.shape[-1]
    embedding = np.zeros((len(adj_loc_list), feature_dim + 128), np.float32)
    for a, adj_dict in enumerate(adj_loc_list):
        if a == skip_loc:# the embedding for the first action ('stop') is left as zero
            continue
        embedding[a, :feature_dim] = features[adj_dict['absViewIndex']]
        loc_embedding = embedding[a, feature_dim:]
        rel_heading = adj_dict['rel_heading']
        rel_elevation = adj_dict['rel_elevation']
        loc_embedding[0:32] = np.sin(rel_heading)
        loc_embedding[32:64] = np.cos(rel_heading)
        loc_embedding[64:96] = np.sin(rel_elevation)
        loc_embedding[96:] = np.cos(rel_elevation)
    return embedding


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, nav_graphs, panoramic, action_space, beam_size=1, batch_size=100, seed=10, splits=['train'], tokenizer=None, path_type=None, history=None, blind=False):  # , subgoal
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size, beam_size=beam_size,blind=blind)

        self.data = []
        self.scans = []
        self.panoramic = panoramic
        self.nav_graphs= nav_graphs
        self.action_space = action_space
        self.ctrl_feature = None
        if tokenizer:
            tokname = tokenizer.__class__.__name__

        longest_inst = list()
        longest_ep_len = list()
        for item in load_datasets(splits):

            # For every dialog history, stitch together a single instruction string.
            self.scans.append(item['scan'])
            new_item = dict(item)
            new_item['inst_idx'] = item['inst_idx']
            if history == 'none':  # no language input at all
                new_item['instructions'] = ''
                if tokenizer:
                    new_item['instr_encoding'] = tokenizer.encode_sentence('')
            elif history == 'target' or len(item['dialog_history']) == 0:  # Have to use target only if no dialog history.
                tar = item['target']
                new_item['instructions'] = '<TAR> ' + tar
                if tokenizer:
                    if tokname == 'Tokenizer':
                        new_item['instr_encoding'] = tokenizer.encode_sentence([tar], seps=['<TAR>'])
                    else:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(tar)

            elif history == 'oracle_ans':
                ora_a = item['dialog_history'][-1]['message']  # i.e., the last oracle utterance.
                tar = item['target']
                new_item['instructions'] = '<ORA> ' + ora_a + ' <TAR> ' + tar
                if tokenizer:
                    if tokname == 'Tokenizer':
                        new_item['instr_encoding'] = tokenizer.encode_sentence([ora_a, tar], seps=['<ORA>', '<TAR>'])
                    else:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(new_item['instructions'])

            elif history == 'nav_q_oracle_ans':
                nav_q = item['dialog_history'][-2]['message']
                ora_a = item['dialog_history'][-1]['message']
                tar = item['target']
                new_item['instructions'] = '<NAV> ' + nav_q + ' <ORA> ' + ora_a + ' <TAR> ' + tar
                if tokenizer:
                    if tokname == 'Tokenizer':
                        qa_enc = tokenizer.encode_sentence([nav_q, ora_a, tar], seps=['<NAV>', '<ORA>', '<TAR>'])
                    else:
                        qa_enc = tokenizer.encode_sentence(new_item['instructions'])
                    new_item['instr_encoding'] = qa_enc
            elif history == 'all':
                dia_inst = ''
                sentences = []
                seps = []
                for turn in item['dialog_history']:
                    sentences.append(turn['message'])
                    sep = '<NAV>' if turn['role'] == 'navigator' else '<ORA>'
                    seps.append(sep)
                    dia_inst += sep + ' ' + turn['message'] + ' '
                sentences.append(item['target'])
                seps.append('<TAR>')
                dia_inst += '<TAR> ' + item['target']
                new_item['instructions'] = dia_inst
                if tokenizer:
                    if tokname == "Tokenizer":
                        dia_enc = tokenizer.encode_sentence(sentences, seps=seps)
                    else:
                        dia_enc = tokenizer.encode_sentence(dia_inst)
                    new_item['instr_encoding'] = dia_enc

            # If evaluating against 'trusted_path', we need to calculate the trusted path and instantiate it.
            if path_type == 'trusted_path':
                # The trusted path is either the planner_path or the player_path depending on whether the player_path
                # contains the planner_path goal (e.g., stricter planner oracle success of player_path
                # indicates we can 'trust' it, otherwise we fall back to the planner path for supervision).
                # Hypothesize that this will combine the strengths of good human exploration with the known good, if
                # short, routes the planner uses.
                planner_goal = item['planner_path'][-1]  # this could be length 1 if "plan" is to not move at all.
                if planner_goal in item['player_path'][1:]:  # player walked through planner goal (did not start on it)
                    new_item['trusted_path'] = item['player_path'][:]  # trust the player.
                else:
                    new_item['trusted_path'] = item['planner_path'][:]  # trust the planner.
					
            longest_ep_len.append(len(new_item[path_type]))
            longest_inst.append(len(new_item['instructions'].split()))

            self.data.append(new_item)


        self.scans = set(self.scans)
        self.splits = splits
        if seed!= 'resume':
            self.seed = seed
            random.seed(self.seed)
            random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        self.epo_inc = False  # jolin: middle of an epoch
        self.path_type = path_type

        #self.env.pre_loadSims(self.data) # debug
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))
        print('Instructions avg length %d, max length %d, using splits: %s' % (np.mean(longest_inst),np.max(longest_inst), ",".join(splits)))
        print('Path avg length %d, max length %d, using splits: %s' % (np.mean(longest_ep_len),np.max(longest_ep_len), ",".join(splits)))

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))


    def _next_minibatch(self,sort=False):
        batch = self.data[self.ix: self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
            self.epo_inc = True  # jolin: end of an epoch
        else:
            self.ix += self.batch_size
            self.epo_inc = False  # jolin: middle of an epoch

        if sort:
            batch = sorted(batch, key=lambda item: np.argmax(item['instr_encoding']==padding_idx), reverse=True)
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0
        self.epo_inc = False  # jolin: middle of an epoch

    def _shortest_path_pano_action(self, state, adj_loc_list, goalViewpointId):
        '''
                Determine next action on the shortest path to goal,
                for supervised training.
                '''
        if state.location.viewpointId == goalViewpointId:
            return 0  # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        for n_a, loc_attr in enumerate(adj_loc_list):
            if loc_attr['nextViewpointId'] == nextViewpointId:
                return n_a

        # Next nextViewpointId not found! This should not happen!
        print('adj_loc_list:', adj_loc_list)
        print('nextViewpointId:', nextViewpointId)
        long_id = '{}_{}'.format(state.scanId, state.location.viewpointId)
        print('longId:', long_id)
        raise Exception('Bug: nextViewpointId not in adj_loc_list')

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return (0, 0, 0) # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        # Can we see the next viewpoint?
        for i,loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0) # Turn right
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0) # Turn left
                elif loc.rel_elevation > math.pi/6.0 and state.viewIndex//12 < 2:
                      return (0, 0, 1) # Look up
                elif loc.rel_elevation < -math.pi/6.0 and state.viewIndex//12 > 0:
                      return (0, 0,-1) # Look down
                else:
                      return (i, 0, 0) # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex//12 == 0:
            return (0, 0, 1) # Look up
        elif state.viewIndex//12 == 2:
            return (0, 0,-1) # Look down
        # Otherwise decide which way to turn
        target_rel = self.graphs[state.scanId].node[nextViewpointId]['position'] - state.location.point
        target_heading = math.pi/2.0 - math.atan2(target_rel[1], target_rel[0]) # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0*math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0,-1, 0) # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0,-1, 0) # Turn left
        return (0, 1, 0) # Turn right


    def _get_obs_inside(self, item, feature, state, sim):
        action_embedding, adj_loc_list = None, None
        if self.panoramic:
            feature_all, feature_1 = feature
            feature_with_loc_all = np.concatenate((feature_all, _static_loc_embeddings[state.viewIndex]), axis=-1)
            feature = (feature_with_loc_all, feature_1)

            if self.action_space == -1:
                adj_loc_list = _get_panorama_states(state, sim, self.nav_graphs)
                action_embedding = _build_action_embedding(adj_loc_list, feature_all)
        ctrl_features = []

        if self.action_space == 6:  # todo: add discrete space single ctrl feature
            teacher = self._shortest_path_action(state, item[self.path_type][-1])
        else:  # navigable
            teacher = self._shortest_path_pano_action(state, adj_loc_list, item[self.path_type][-1])

        ret = {
            'inst_idx': item['inst_idx'],
            'scan': state.scanId,
            'viewpoint': state.location.viewpointId,
            'viewIndex': state.viewIndex,
            'heading': state.heading,
            'elevation': state.elevation,
            'feature': feature,
            'step': state.step,
            'adj_loc_list': adj_loc_list,
            'action_embedding': action_embedding,
            'navigableLocations': state.navigableLocations,
            'instructions': item['instructions'],
            'teacher': teacher,
            'ctrl_features': ctrl_features
        }

        if 'instr_encoding' in item:
            ret['instr_encoding'] = item['instr_encoding']
        return ret

    def _get_obs(self, all_states):#, sub_stages):  # sub_stages: how many stages left
        obs = []
        for i,(states) in enumerate(all_states):
            item = self.batch[i]
            feature, state, sim = states
            obs.append(self._get_obs_inside(item,feature,state,sim))
        #if isinstance(all_states[0], tuple):
        #    for i,(states) in enumerate(all_states):
        #        item = self.batch[i]
        #        feature, state, sim = states
        #        obs.append(self._get_obs_inside(item,feature,state,sim))
        #else:# beamed
        #    for i,(sims, beam_states) in enumerate(zip(self.env.sims_view(), all_states)):
        #        item = self.batch[i]
        #        beam_obs = []
        #        for sim, (feature, world_state) in zip(sims, beam_states):
        #            sim.newEpisode(*world_state)
        #            beam_obs.append(self._get_obs_inside(item,feature,sim.getState(),sim))
        #        obs.append(beam_obs)
        return obs

    def reset(self, sort):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(sort)
        #scanIds = [item['scan'] for item in self.batch]
        #viewpointIds = [item['path'][0] for item in self.batch]
        #headings = [item['heading'] for item in self.batch]
        #all_states = self.env.newEpisodes(scanIds, viewpointIds, headings)

        #svhs = [(item['scan'], item['path'][0], item['heading']) for item in self.batch]
        svhs = [(item['scan'], item[self.path_type][0], item['start_pano']['heading']) for item in self.batch]
        all_states = self.env.newBatchEpisodes(svhs)

        #pathIds = [item['path_id'] for item in self.batch]
        #all_states = self.env.batchEpisodes(pathIds, scanIds, viewpointIds, headings)
        return all_states

    def reset_batch(self):
        ''' jolin: Load last minibatch / episodes. '''
        #scanIds = [item['scan'] for item in self.batch]
        #viewpointIds = [item['path'][0] for item in self.batch]
        #headings = [item['heading'] for item in self.batch]
        #all_states = self.env.newEpisodes(scanIds, viewpointIds, headings)

        svhs = [(item['scan'], item[self.path_type][0], item['start_pano']['heading']) for item in self.batch]
        all_states = self.env.newBatchEpisodes(svhs)

        #pathIds = [item['path_id'] for item in self.batch]
        #all_states = self.env.batchEpisodes(pathIds, scanIds, viewpointIds, headings)
        return all_states

    def step(self, actions, last_obs=None, world_states=None):
        ''' Take action (same interface as makeActions) '''
        if self.action_space == 6:
            self.env.makeActions(actions)
            all_states = self.env.getStates()
        else:
            if world_states is None:
                for i, (sim, action, last_ob) in enumerate(zip(self.env.sims, actions, last_obs)):
                    loc_attr = last_ob['adj_loc_list'][action]
                    _navigate_to_location(sim, loc_attr['nextViewpointId'], loc_attr['absViewIndex'])
                all_states = self.env.getStates()
            else:  # beamed
                all_states = []
                for t in zip(self.env.sims_view(), world_states, actions, last_obs):
                    beam_states = []
                    for sim, state, action, last_ob in zip(*t):
                        sim.newEpisode(*state)
                        loc_attr = last_ob['adj_loc_list'][action]
                        _navigate_to_location(sim, loc_attr['nextViewpointId'], loc_attr['absViewIndex'])
                        new_state = sim.getState()
                        feature = self.env.feature.rollout(new_state.scanId,
                                                       new_state.location.viewpointId,
                                                       new_state.viewIndex)
                        world_state = WorldState(scanId=new_state.scanId,
                                   viewpointId=new_state.location.viewpointId,
                                   heading=new_state.heading,
                                   elevation=new_state.elevation)
                        beam_states.append((feature, world_state))
                    all_states.append(beam_states)
        return all_states


    def world_states2feature_states(self, world_states):  # only for state_factored_search
        all_states=[]
        for sims, beam in zip(self.env.sims_view(), world_states):
            beam_states = []
            for sim, world_state in zip(sims, beam):
                sim.newEpisode(*world_state)
                new_state = sim.getState()
                feature = self.env.feature.rollout(new_state.scanId,
                                               new_state.location.viewpointId,
                                               new_state.viewIndex)
                beam_states.append((feature, world_state))
            all_states.append(beam_states)
        return all_states




class OR2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 path_type='planner_path', history='target', blind=False):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size, blind=blind)
        self.data = []
        self.scans = []
        if tokenizer:
            tokname = tokenizer.__class__.__name__
        for item in load_datasets(splits):

            # For every dialog history, stitch together a single instruction string.
            self.scans.append(item['scan'])
            new_item = dict(item)
            new_item['inst_idx'] = item['inst_idx']
            if history == 'none':  # no language input at all
                new_item['instructions'] = ''
                if tokenizer:
                    new_item['instr_encoding'] = tokenizer.encode_sentence('')
            elif history == 'target' or len(item['dialog_history']) == 0:  # Have to use target only if no dialog history.
                tar = item['target']
                new_item['instructions'] = '<TAR> ' + tar
                if tokenizer:
                    if tokname == 'Tokenizer':
                        new_item['instr_encoding'] = tokenizer.encode_sentence([tar], seps=['<TAR>'])
                    else:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(tar)

            elif history == 'oracle_ans':
                ora_a = item['dialog_history'][-1]['message']  # i.e., the last oracle utterance.
                tar = item['target']
                new_item['instructions'] = '<ORA> ' + ora_a + ' <TAR> ' + tar
                if tokenizer:
                    new_item['instr_encoding'] = tokenizer.encode_sentence([ora_a, tar], seps=['<ORA>', '<TAR>'])
            elif history == 'nav_q_oracle_ans':
                nav_q = item['dialog_history'][-2]['message']
                ora_a = item['dialog_history'][-1]['message']
                tar = item['target']
                new_item['instructions'] = '<NAV> ' + nav_q + ' <ORA> ' + ora_a + ' <TAR> ' + tar
                if tokenizer:
                    qa_enc = tokenizer.encode_sentence([nav_q, ora_a, tar], seps=['<NAV>', '<ORA>', '<TAR>'])
                    new_item['instr_encoding'] = qa_enc
            elif history == 'all':
                dia_inst = ''
                sentences = []
                seps = []
                for turn in item['dialog_history']:
                    sentences.append(turn['message'])
                    sep = '<NAV>' if turn['role'] == 'navigator' else '<ORA>'
                    seps.append(sep)
                    dia_inst += sep + ' ' + turn['message'] + ' '
                sentences.append(item['target'])
                seps.append('<TAR>')
                dia_inst += '<TAR> ' + item['target']
                new_item['instructions'] = dia_inst
                if tokenizer:
                    if tokname == "Tokenizer":
                        dia_enc = tokenizer.encode_sentence(sentences, seps=seps)
                    else:
                        dia_enc = tokenizer.encode_sentence(dia_inst)
                    new_item['instr_encoding'] = dia_enc

            # If evaluating against 'trusted_path', we need to calculate the trusted path and instantiate it.
            if path_type == 'trusted_path':
                # The trusted path is either the planner_path or the player_path depending on whether the player_path
                # contains the planner_path goal (e.g., stricter planner oracle success of player_path
                # indicates we can 'trust' it, otherwise we fall back to the planner path for supervision).
                # Hypothesize that this will combine the strengths of good human exploration with the known good, if
                # short, routes the planner uses.
                planner_goal = item['planner_path'][-1]  # this could be length 1 if "plan" is to not move at all.
                if planner_goal in item['player_path'][1:]:  # player walked through planner goal (did not start on it)
                    new_item['trusted_path'] = item['player_path'][:]  # trust the player.
                else:
                    new_item['trusted_path'] = item['planner_path'][:]  # trust the planner.

            self.data.append(new_item)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        self.path_type = path_type
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def _load_nav_graphs(self):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return (0, 0, 0) # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        # Can we see the next viewpoint?
        for i,loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextViewpointId:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0) # Turn right
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0) # Turn left
                elif loc.rel_elevation > math.pi/6.0 and state.viewIndex//12 < 2:
                      return (0, 0, 1) # Look up
                elif loc.rel_elevation < -math.pi/6.0 and state.viewIndex//12 > 0:
                      return (0, 0,-1) # Look down
                else:
                      return (i, 0, 0) # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex//12 == 0:
            return (0, 0, 1) # Look up
        elif state.viewIndex//12 == 2:
            return (0, 0,-1) # Look down
        # Otherwise decide which way to turn
        #pos = [state.location.x, state.location.y, state.location.z]
        pos = state.location.point
        target_rel = self.graphs[state.scanId].node[nextViewpointId]['position'] - pos
        target_heading = math.pi/2.0 - math.atan2(target_rel[1], target_rel[0]) # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0*math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return (0,-1, 0) # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return (0,-1, 0) # Turn left
        return (0, 1, 0) # Turn right

    def _get_obs(self):
        obs = []
        #for i,(feature,state) in enumerate(self.env.getStates()):
        for i,(feature,state, si) in enumerate(self.env.getStates()):
            item = self.batch[i]
            obs.append({
                'inst_idx': item['inst_idx'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'step': state.step,
                'navigableLocations': state.navigableLocations,
                'instructions': item['instructions'],
                'teacher': self._shortest_path_action(state, item[self.path_type][-1]),
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
        return obs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch()
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item[self.path_type][0] for item in self.batch]
        headings = [item['start_pano']['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()


