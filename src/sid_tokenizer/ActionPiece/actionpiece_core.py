"""
Core ActionPiece tokenizer implementation.

This is a standalone copy of the ActionPiece core algorithm,
independent of the original genrec package.

Based on the paper: "ActionPiece: Contextually Tokenizing Action Sequences 
for Generative Recommendation"
"""

import collections
import heapq
import json
import random
from typing import Any, List, Dict, Optional

import numpy as np
import tqdm

tqdm = tqdm.tqdm


class LinkedListState:
    """Node of the linked list for ActionPiece tokenization.
    
    Attributes:
        state: The state of the node (list of token IDs).
        head_id: The head id of the node.
        context: Whether the node is a context slot.
        next: The next node of the linked list.
        prev: The previous node of the linked list.
    """

    def __init__(self, state: List[int], head_id: int, context: bool):
        self.state = state
        self.head_id = head_id
        self.context = context
        self.next = None
        self.prev = None

    def append(self, node: "LinkedListState") -> "LinkedListState":
        """Append a node to the end of the linked list."""
        self.next = node
        node.prev = self
        return self.next

    def copy(self) -> "LinkedListState":
        """Duplicate this node."""
        new_node = LinkedListState(
            state=self.state.copy(),
            head_id=self.head_id,
            context=self.context,
        )
        new_node.next = self.next
        new_node.prev = self.prev
        return new_node

    def copy_link(self) -> "LinkedListState":
        """Duplicate this node and its following nodes."""
        new_link = LinkedListState(
            state=self.state.copy(),
            head_id=self.head_id,
            context=self.context,
        )
        old_node = self.next
        cur_node = new_link
        while old_node:
            new_node = LinkedListState(
                state=old_node.state.copy(),
                head_id=old_node.head_id,
                context=old_node.context,
            )
            cur_node.next = new_node
            new_node.prev = cur_node
            old_node = old_node.next
            cur_node = new_node
        return new_link

    def nextk(self, k: int) -> "LinkedListState":
        """Return the k-th next node of the linked list."""
        cur_node = self
        for index in range(k):
            if not cur_node.next:
                print("Invalid k, maximum k is ", index)
                break
            cur_node = cur_node.next
        return cur_node

    def tolist(self) -> List[int]:
        """Convert the linked list to a list of states."""
        cur_node = self
        res = []
        while cur_node:
            res.extend(cur_node.state)
            cur_node = cur_node.next
        return res

    def to_shuffled_list(self) -> List[int]:
        """Convert the linked list to a list of states where each state is shuffled."""
        cur_node = self
        res = []
        while cur_node:
            dup_state = cur_node.state.copy()
            random.shuffle(dup_state)
            res.extend(dup_state)
            cur_node = cur_node.next
        return res

    def __str__(self):
        return f"head_id: {self.head_id}, state: {self.state}, context: {self.context}"


def diff_cnt(cnt1: Dict, cnt2: Dict) -> Dict:
    """Minus the second pair2cnt from the first pair2cnt."""
    return {k: v - cnt2.get(k, 0) for k, v in cnt1.items()}


def add_cnt_inplace(cnt1: Dict, cnt2: Dict):
    """Add pair2cnt inplace."""
    for k, v in cnt2.items():
        cnt1[k] += v


class ActionPieceCore:
    """The core ActionPiece tokenizer.
    
    This class can be initialized in three ways:
    1. From state2feat: dict mapping state (str) to features (list[int]).
    2. From metadata: dict containing the metadata of a trained ActionPiece.
    3. Using from_pretrained(): Load from a saved metadata file.
    
    Attributes:
        vocab: The vocabulary of ActionPiece (same as token2feat).
        rank: The rank of tokens (same as feat2token).
        vocab_size: The size of the vocabulary.
        n_categories: The number of categories of the features.
        priority: The priority of each token.
    """

    def __init__(self, state2feat: Optional[Dict] = None, metadata: Optional[Dict] = None):
        self.state2feat = state2feat
        self.metadata = metadata
        self.token2all_feat = {}

        if self.state2feat is not None:
            self.n_categories, self.token2feat, self.feat2token, self.priority = (
                self._init_from_state2feat(state2feat)
            )
            self.n_init_feats = len(self.token2feat)
        elif metadata is not None:
            (
                self.n_categories,
                self.n_init_feats,
                self.token2feat,
                self.feat2token,
                self.priority,
            ) = self._init_from_metadata(metadata)
        else:
            raise ValueError("Check that one of state2feat and metadata is None.")
        self.eps = 1e-12

    @property
    def vocab(self):
        return self.token2feat

    @property
    def rank(self):
        return self.feat2token

    @property
    def vocab_size(self):
        return len(self.token2feat)

    def _init_from_state2feat(self, state2feat: Dict[str, List[int]]):
        """Initialize using the most basic features from state2feat."""
        vocab = [(-1, -1)]  # The first token is the padding token
        rank = {(-1, -1): 0}
        priority = [0]
        feats = np.array(list(state2feat.values()))
        for i in range(feats.shape[1]):
            for j in np.unique(feats[:, i]).tolist():
                rank[(i, j)] = len(vocab)
                vocab.append((i, j))
                priority.append(0)
        return feats.shape[1], vocab, rank, priority

    def _init_from_metadata(self, metadata: Dict[str, Any]):
        """Initialize ActionPiece from the metadata of a trained ActionPiece."""
        n_categories = metadata["n_categories"]
        n_init_feats = metadata["n_init_feats"]
        token2feat = [tuple(_) for _ in metadata["token2feat"]]
        feat2token = {feat: token for token, feat in enumerate(token2feat)}
        priority = [float(_) for _ in metadata["priority"]]
        return n_categories, n_init_feats, token2feat, feat2token, priority

    def save(self, save_path: str):
        """Save ActionPiece to a metadata file."""
        data = {
            "n_categories": self.n_categories,
            "n_init_feats": self.n_init_feats,
            "token2feat": self.token2feat,
            "priority": self.priority,
        }
        with open(save_path, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_pretrained(cls, save_path: str, vocab_size: Optional[int] = None):
        """Initialize ActionPiece from a saved file."""
        with open(save_path, "r") as f:
            metadata = json.load(f)
        if vocab_size is not None:
            assert vocab_size >= metadata["n_init_feats"], (
                f"The target vocab size ({vocab_size}) must be larger than the"
                f' initial vocab size ({metadata["n_init_feats"]})'
            )
            assert vocab_size <= len(metadata["token2feat"]), (
                f"The target vocab size ({vocab_size}) must be smaller than the"
                f' number of tokens ({len(metadata["token2feat"])})'
            )
            metadata["token2feat"] = metadata["token2feat"][:vocab_size]
            metadata["priority"] = metadata["priority"][:vocab_size]
        actionpiece = cls(metadata=metadata)
        return actionpiece

    def _construct_linked_list(self, head_id: int, state_seq) -> LinkedListState:
        """Construct the linked list for a single state sequence."""
        state_seq = state_seq.tolist()
        head = LinkedListState(state_seq[0], head_id, context=False)
        tail = head
        for state in state_seq[1:]:
            tail = tail.append(LinkedListState([], head_id, context=True))
            tail = tail.append(LinkedListState(state, head_id, context=False))
        return head

    def _count_pairs_inside_state(self, state: List[int]) -> Dict:
        """Count the pairs of tokens inside a single state."""
        pair2cnt = collections.defaultdict(float)
        for p, tk1 in enumerate(state):
            for tk2 in state[p + 1 :]:
                pair2cnt[(min(tk1, tk2), max(tk1, tk2))] += 2 / len(state)
        return pair2cnt

    def _count_pairs_btw_states(self, state1: List[int], state2: List[int]) -> Dict:
        """Iterate all the pairs of tokens between two states."""
        pair2cnt = collections.defaultdict(float)
        for tk1 in state1:
            for tk2 in state2:
                pair2cnt[(min(tk1, tk2), max(tk1, tk2))] += 1 / (len(state1) * len(state2))
        return pair2cnt

    def _count_pairs_in_list(self, head: LinkedListState) -> Dict:
        """Count the pairs of tokens in a single linked list."""
        pair2cnt = collections.defaultdict(float)
        cur_node = head
        while cur_node:
            add_cnt_inplace(pair2cnt, self._count_pairs_inside_state(cur_node.state))
            if not cur_node.next:
                break
            if cur_node.next.state:
                add_cnt_inplace(
                    pair2cnt,
                    self._count_pairs_btw_states(cur_node.state, cur_node.next.state),
                )
                add_cnt_inplace(
                    pair2cnt,
                    self._count_pairs_btw_states(cur_node.next.state, cur_node.next.next.state),
                )
            else:
                add_cnt_inplace(
                    pair2cnt,
                    self._count_pairs_btw_states(cur_node.state, cur_node.next.next.state),
                )
            cur_node = cur_node.next.next
        return pair2cnt

    def _build(self, token_corpus):
        """Build the data structures for the training process."""
        self.cur_corpus = [
            self._construct_linked_list(i, state_seq)
            for i, state_seq in enumerate(token_corpus)
        ]
        self.head_id2pair_cnt = []
        self.pair2head_ids = collections.defaultdict(set)
        self.all_pair2cnt = collections.defaultdict(float)
        
        for head in self.cur_corpus:
            head_id = head.head_id
            pair2cnt = self._count_pairs_in_list(head)
            add_cnt_inplace(self.all_pair2cnt, pair2cnt)
            for pair in pair2cnt:
                self.pair2head_ids[pair].add(head_id)
            self.head_id2pair_cnt.append(pair2cnt)

        # Use heapq for efficient lazy updates (Appendix C optimization)
        self.pq = []
        for (tk1, tk2), cnt in self.all_pair2cnt.items():
            heapq.heappush(self.pq, (-cnt, (tk1, tk2)))

    def _outdated(self, pair, priority) -> bool:
        """Check if the pair is outdated in the priority queue."""
        return abs(priority - self.all_pair2cnt[pair]) > self.eps

    def _merge_empty_nodes(self, head: LinkedListState) -> LinkedListState:
        """Merge empty nodes in the linked list."""
        updated_flag = True
        while updated_flag:
            updated_flag = False
            cur_node = head
            while cur_node:
                if cur_node.context:
                    cur_node = cur_node.next
                    continue
                if not cur_node.state and cur_node.prev:
                    if cur_node.prev.context and cur_node.prev.state:
                        cur_node.state = cur_node.prev.state
                        cur_node.prev.state = []
                        updated_flag = True
                cur_node = cur_node.next
            cur_node = head
            while cur_node:
                if cur_node.context:
                    cur_node = cur_node.next
                    continue
                if not cur_node.next:
                    break
                if not cur_node.state and not cur_node.next.state:
                    if not cur_node.prev:
                        head = cur_node.next.next
                        cur_node.next.next.prev = None
                    else:
                        cur_node.prev.next = cur_node.next.next
                        cur_node.next.next.prev = cur_node.prev
                    updated_flag = True
                    cur_node = cur_node.next.next
                else:
                    cur_node = cur_node.next
        return head

    def _merge_inside_regular_state(self, node, rule, new_token):
        """Merge the tokens inside a regular state."""
        if rule[0] == rule[1]:
            return
        if rule[0] in node.state and rule[1] in node.state:
            node.state = [new_token] + [state for state in node.state if state not in rule]

    def _merge_state_context(self, state_node, context_node, rule, new_token):
        """Merge the tokens between a regular state and a context slot."""
        assert len(context_node.state) == 1
        if rule[0] == context_node.state[0]:
            if rule[1] in state_node.state:
                state_node.state = [_ for _ in state_node.state if _ != rule[1]]
                context_node.state = [new_token]
        elif rule[1] == context_node.state[0]:
            if rule[0] in state_node.state:
                state_node.state = [_ for _ in state_node.state if _ != rule[0]]
                context_node.state = [new_token]

    def _merge_two_states(self, node1, node2, rule, new_token):
        """Merge the tokens between two regular states."""
        assert not node1.next.state
        if rule[0] in node1.state and rule[1] in node2.state:
            node1.state = [item for item in node1.state if item != rule[0]]
            node2.state = [item for item in node2.state if item != rule[1]]
            node1.next.state = [new_token]
        elif rule[1] in node1.state and rule[0] in node2.state:
            node1.state = [item for item in node1.state if item != rule[1]]
            node2.state = [item for item in node2.state if item != rule[0]]
            node2.prev.state = [new_token]

    def _merge_single_rule(self, head, rule, new_token) -> LinkedListState:
        """Merge the tokens in the linked list according to the new merging rule."""
        new_link = head.copy_link()
        cur_node = new_link
        while cur_node:
            assert not cur_node.context, "cur_node should be a regular state"
            self._merge_inside_regular_state(cur_node, rule, new_token)
            if not cur_node.next:
                break
            if cur_node.next.state:
                self._merge_state_context(cur_node, cur_node.next, rule, new_token)
                self._merge_state_context(cur_node.next.next, cur_node.next, rule, new_token)
            else:
                self._merge_two_states(cur_node, cur_node.next.next, rule, new_token)
            cur_node = cur_node.next.next
        return self._merge_empty_nodes(new_link)

    def _update_pair2head_ids(self, diff_pair2cnt, head_id):
        """Update the inverted index based on the diff of pair counting."""
        for pair in diff_pair2cnt:
            if diff_pair2cnt[pair] > 0 and abs(self.head_id2pair_cnt[head_id][pair]) < self.eps:
                assert head_id not in self.pair2head_ids[pair]
                self.pair2head_ids[pair].add(head_id)
            elif (
                diff_pair2cnt[pair] < 0
                and abs(self.head_id2pair_cnt[head_id][pair] + diff_pair2cnt[pair]) < self.eps
            ):
                assert head_id in self.pair2head_ids[pair]
                self.pair2head_ids[pair].remove(head_id)

    def _update_pq(self, diff):
        """Update the priority queue using the lazy update strategy."""
        for pair in diff:
            if abs(diff[pair]) < self.eps:
                continue
            self.all_pair2cnt[pair] += diff[pair]
            # Lazy update: just push new entry, outdated check handles duplicates
            heapq.heappush(self.pq, (-self.all_pair2cnt[pair], pair))

    def _get_token_corpus(self, state_corpus):
        """Get the token corpus from the state corpus."""
        token_corpus = []
        for state_seq in state_corpus:
            token_seq = [
                [self.feat2token[it] for it in enumerate(self.state2feat[state])]
                for state in state_seq
            ]
            token_corpus.append(np.array(token_seq))
        return token_corpus

    def train(self, state_corpus, target_vocab_size: int):
        """Train the ActionPiece tokenizer."""
        token_corpus = self._get_token_corpus(state_corpus)
        self._build(token_corpus)

        progress_bar = tqdm(range(target_vocab_size - self.n_init_feats))
        while len(self.vocab) < target_vocab_size:
            self._train_step()
            progress_bar.set_description(f"[Vocab size: {len(self.vocab)} / {target_vocab_size}] ")
            progress_bar.update(1)
        progress_bar.close()

    def _train_step(self):
        """Single training step: merge the most frequent pair."""
        priority, tk1, tk2 = None, None, None
        # Pop from heap until we find a valid (non-outdated) pair
        while self.pq:
            priority, (tk1, tk2) = heapq.heappop(self.pq)
            if not self._outdated((tk1, tk2), -priority):
                break

        new_rule = (-1, tk1, tk2)
        new_token = len(self.vocab)
        self.rank[new_rule] = new_token
        self.vocab.append(new_rule)
        self.priority.append(-priority)

        head_to_update = self.pair2head_ids[(tk1, tk2)].copy()
        all_diff = collections.defaultdict(int)
        
        for head_id in head_to_update:
            self.cur_corpus[head_id] = self._merge_single_rule(
                self.cur_corpus[head_id], rule=(tk1, tk2), new_token=new_token
            )
            new_pair2cnt = self._count_pairs_in_list(self.cur_corpus[head_id])
            diff_pair2cnt = diff_cnt(new_pair2cnt, self.head_id2pair_cnt[head_id])
            self._update_pair2head_ids(diff_pair2cnt, head_id)
            self.head_id2pair_cnt[head_id] = new_pair2cnt
            add_cnt_inplace(all_diff, diff_pair2cnt)
        self._update_pq(all_diff)

    def _random_walk_augmentation(self, state_seq: np.ndarray) -> List[int]:
        """Random walk augmentation (SPR - Set Permutation Regularization)."""
        aug_state_seq = []
        for seq in state_seq:
            aug_state_seq.extend(np.random.permutation(seq).tolist())
        return aug_state_seq

    def _encode(self, seq: List[int]) -> List[int]:
        """Encode a flattened feature sequence into a token sequence (BPE-style)."""
        while True:
            min_idx = None
            min_rank = float("inf")
            for i, (tk1, tk2) in enumerate(zip(seq[:-1], seq[1:])):
                tk1, tk2 = min(tk1, tk2), max(tk1, tk2)
                cur_rank = self.rank.get((-1, tk1, tk2))
                if cur_rank is not None and cur_rank < min_rank:
                    min_idx = i
                    min_rank = cur_rank
            if min_idx is None:
                break
            seq = seq[:min_idx] + [min_rank] + seq[min_idx + 2 :]
        return seq

    def encode_fast(self, state_seq: np.ndarray) -> List[int]:
        """Fast encoding with random walk augmentation."""
        aug_state_seq = self._random_walk_augmentation(state_seq)
        return self._encode(aug_state_seq)

    def encode(self, state_seq: np.ndarray, shuffle: str = "feature") -> List[int]:
        """Encode the state sequence into a list of tokens.
        
        Args:
            state_seq: The state sequence (N, n_categories).
            shuffle: Shuffle strategy:
                - 'feature': random walk augmentation (SPR)
                - 'token': enumerate pairs, merge, shuffle inside state
                - 'none': enumerate pairs, merge, no shuffle
        
        Returns:
            Encoded token sequence.
        """

        def _count_inside_ll(node, updates):
            best_priority, node_to_update, rule_to_update = updates
            for i, tk1 in enumerate(node.state):
                for tk2 in node.state[i + 1 :]:
                    cur_rule = (-1, min(tk1, tk2), max(tk1, tk2))
                    if cur_rule not in self.rank:
                        continue
                    score = self.priority[self.rank[cur_rule]] * 2 / len(node.state)
                    if best_priority is None or score > best_priority:
                        best_priority = score
                        node_to_update = (node,)
                        rule_to_update = cur_rule
            return (best_priority, node_to_update, rule_to_update)

        def _count_two_states_ll(node1, node2, updates):
            best_priority, node_to_update, rule_to_update = updates
            for tk1 in node1.state:
                for tk2 in node2.state:
                    cur_rule = (-1, min(tk1, tk2), max(tk1, tk2))
                    if cur_rule not in self.rank:
                        continue
                    score = self.priority[self.rank[cur_rule]] / (len(node1.state) * len(node2.state))
                    if best_priority is None or score > best_priority:
                        best_priority = score
                        node_to_update = (node1, node2)
                        rule_to_update = cur_rule
            return (best_priority, node_to_update, rule_to_update)

        if shuffle == "feature":
            return self.encode_fast(state_seq)
        else:
            head = self._construct_linked_list(head_id=-1, state_seq=state_seq)
            while True:
                cur_updates = (None, None, None)
                cur_node = head
                while cur_node:
                    cur_updates = _count_inside_ll(cur_node, cur_updates)
                    if not cur_node.next:
                        break
                    if cur_node.next.state:
                        cur_updates = _count_two_states_ll(cur_node, cur_node.next, cur_updates)
                        cur_updates = _count_two_states_ll(cur_node.next.next, cur_node.next, cur_updates)
                    else:
                        cur_updates = _count_two_states_ll(cur_node, cur_node.next.next, cur_updates)
                    cur_node = cur_node.next.next
                if cur_updates[0] is None:
                    break
                _, node_to_update, rule_to_update = cur_updates
                if len(node_to_update) == 1:
                    self._merge_inside_regular_state(
                        node_to_update[0],
                        (rule_to_update[1], rule_to_update[2]),
                        new_token=self.rank[rule_to_update],
                    )
                else:
                    if node_to_update[1].context:
                        self._merge_state_context(
                            node_to_update[0],
                            node_to_update[1],
                            (rule_to_update[1], rule_to_update[2]),
                            new_token=self.rank[rule_to_update],
                        )
                    else:
                        self._merge_two_states(
                            node_to_update[0],
                            node_to_update[1],
                            (rule_to_update[1], rule_to_update[2]),
                            new_token=self.rank[rule_to_update],
                        )
                head = self._merge_empty_nodes(head)
            if shuffle == "token":
                return head.to_shuffled_list()
            elif shuffle == "none":
                return head.tolist()

    def _decode_single_token(self, token: int) -> List[tuple]:
        """Decode a single token into the most basic features."""
        if token in self.token2all_feat:
            return self.token2all_feat[token]
        decoded = self.vocab[token]
        if decoded[0] == -1:
            assert len(decoded) == 3, f"Invalid token: {token}"
            all_feat = self._decode_single_token(decoded[1]) + self._decode_single_token(decoded[2])
        else:
            all_feat = [decoded]
        self.token2all_feat[token] = all_feat
        return all_feat

    def decode_single_state(self, token_seq: List[int]) -> Optional[List[tuple]]:
        """Decode a sequence of tokens into the most basic features.
        
        Args:
            token_seq: The token sequence to decode.
            
        Returns:
            None if invalid, otherwise list of (category_idx, feature_idx) tuples.
        """
        cur_state = {}
        for token in token_seq:
            if token == 0:
                return None
            if token >= len(self.vocab):
                print(f"Invalid token: {token}")
                return None
            feats = self._decode_single_token(token)
            for pos, f in feats:
                if pos in cur_state:
                    return None
                cur_state[pos] = f
        for i in range(self.n_categories):
            if i not in cur_state:
                return None
        return [(i, cur_state[i]) for i in range(self.n_categories)]
