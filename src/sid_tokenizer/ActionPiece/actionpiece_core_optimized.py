"""
Optimized ActionPiece core implementation with Efficient Vocabulary Construction.

Key optimizations from paper Appendix C:
1. Max-Heap with proper lazy update (avoid redundant heap operations)
2. Efficient inverted index updates (only process affected sequences)
3. Early termination for low-frequency pairs
4. Batch processing for large collision groups

This should significantly speed up training on large datasets like Amazon-CDs.
"""

import collections
import heapq
from typing import Dict, Set, Tuple, List
from .actionpiece_core import ActionPieceCore, LinkedListState, add_cnt_inplace, diff_cnt


class ActionPieceCoreOptimized(ActionPieceCore):
    """Optimized ActionPiece with efficient vocabulary construction."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_pair_freq = 1.0  # Skip pairs below this frequency
        
    def _build(self, token_corpus):
        """Build data structures with optimized heap initialization."""
        self.cur_corpus = [
            self._construct_linked_list(i, state_seq)
            for i, state_seq in enumerate(token_corpus)
        ]
        self.head_id2pair_cnt = []
        self.pair2head_ids = collections.defaultdict(set)
        self.all_pair2cnt = collections.defaultdict(float)
        
        # Build inverted index
        for head in self.cur_corpus:
            head_id = head.head_id
            pair2cnt = self._count_pairs_in_list(head)
            add_cnt_inplace(self.all_pair2cnt, pair2cnt)
            for pair in pair2cnt:
                self.pair2head_ids[pair].add(head_id)
            self.head_id2pair_cnt.append(pair2cnt)

        # Use heapq instead of PriorityQueue for better performance
        self.pq = []
        for (tk1, tk2), cnt in self.all_pair2cnt.items():
            if cnt >= self.min_pair_freq:  # Filter low-frequency pairs
                heapq.heappush(self.pq, (-cnt, (tk1, tk2)))
    
    def _update_pq(self, diff):
        """Lazy update: only push new entries, don't remove old ones."""
        for pair in diff:
            if abs(diff[pair]) < self.eps:
                continue
            self.all_pair2cnt[pair] += diff[pair]
            # Only add if frequency is significant
            if self.all_pair2cnt[pair] >= self.min_pair_freq:
                heapq.heappush(self.pq, (-self.all_pair2cnt[pair], pair))
    
    def _train_step(self):
        """Optimized training step with lazy heap updates."""
        # Pop from heap until we find a valid (non-outdated) pair
        priority, tk1, tk2 = None, None, None
        while self.pq:
            priority, (tk1, tk2) = heapq.heappop(self.pq)
            # Check if this entry is outdated (lazy update)
            if not self._outdated((tk1, tk2), -priority):
                break
        
        if priority is None:
            return  # No more pairs to merge
        
        new_rule = (-1, tk1, tk2)
        new_token = len(self.vocab)
        self.rank[new_rule] = new_token
        self.vocab.append(new_rule)
        self.priority.append(-priority)

        # Get sequences to update (inverted index)
        head_to_update = self.pair2head_ids[(tk1, tk2)].copy()
        
        # Batch process updates
        all_diff = collections.defaultdict(int)
        
        # Process in batches to reduce memory pressure for large collision groups
        batch_size = 1000
        for i in range(0, len(head_to_update), batch_size):
            batch = list(head_to_update)[i:i+batch_size]
            
            for head_id in batch:
                # Merge tokens in this sequence
                self.cur_corpus[head_id] = self._merge_single_rule(
                    self.cur_corpus[head_id], rule=(tk1, tk2), new_token=new_token
                )
                
                # Recount pairs in updated sequence
                new_pair2cnt = self._count_pairs_in_list(self.cur_corpus[head_id])
                diff_pair2cnt = diff_cnt(new_pair2cnt, self.head_id2pair_cnt[head_id])
                
                # Update inverted index
                self._update_pair2head_ids(diff_pair2cnt, head_id)
                
                # Update local pair counts
                self.head_id2pair_cnt[head_id] = new_pair2cnt
                
                # Accumulate global diff
                add_cnt_inplace(all_diff, diff_pair2cnt)
        
        # Update priority queue (lazy)
        self._update_pq(all_diff)
