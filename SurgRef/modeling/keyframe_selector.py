"""
Key Frame Selector Module for Surgref
This module implements language-guided key frame selection for efficient
video segmentation. It computes relevance scores for each frame based on
visual-text alignment and selects the most relevant frames for processing.

Author: Meng Wei
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List    


class KeyFrameSelector(nn.Module):
    """
    Language-guided key frame selection module.
    
    For each video clip, the transformer decoder generates language-guided 
    frame-level queries of dimension C_Q, yielding a tensor [T, N_queries, C_Q].
    
    This module:
    1. Aggregates object queries to get frame-level representations [T, C_Q]
    2. Computes scalar relevance scores s_t ∈ [0,1] for each frame using MLP
    3. Selects top-T' frames with highest scores (maintaining temporal order)
    
    Args:
        query_dim (int): Dimension of query features (C_Q). Default: 256
        hidden_dim (int): Hidden dimension for MLP (d). Default: 128
        top_k_ratio (float): Ratio of frames to select. Default: 0.5
        aggregation_method (str): Method to aggregate queries ('mean', 'max', 'attention').
                                 Default: 'mean'
    
    Example:
        selector = KeyFrameSelector(query_dim=256, hidden_dim=128, top_k_ratio=0.5)
        frame_queries = torch.randn(8, 20, 100, 256)  # [B, T, N, C]
        indices, scores = selector(frame_queries)
        print(indices.shape)  # [8, 10] - selected top 50% of 20 frames
    """
    
    def __init__(
        self,
        query_dim: int = 256,
        hidden_dim: int = 128,
        top_k_ratio: float = 0.5,
        aggregation_method: str = 'mean'
    ):
        super().__init__()
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.top_k_ratio = top_k_ratio
        self.aggregation_method = aggregation_method
        
        # Lightweight MLP for scoring: σ(W₂·ReLU(W₁·e_t) + b)
        # where W₁ ∈ ℝ^{d × C_Q}, W₂ ∈ ℝ^{1 × d}
        # self.score_mlp = nn.Sequential(
        #     nn.Linear(query_dim, hidden_dim),  # W₁
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1),  # W₂
        #     nn.Sigmoid()  # σ(·)
        # )

        self.score_mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  # ← Add another layer
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Optional: Attention-based aggregation
        if aggregation_method == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(query_dim, 1),
                nn.Softmax(dim=1)
            )
    
    def aggregate_frame_queries(
        self,
        frame_queries: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate object queries to get frame-level representations e_t.
        
        For each frame I_t, extract its frame-level representation e_t ∈ ℝ^{C_Q}
        by aggregating the corresponding object queries.
        
        Args:
            frame_queries: Tensor of shape [T, N_queries, C_Q]
                          T: number of frames
                          N_queries: number of object queries per frame
                          C_Q: query dimension
        
        Returns:
            frame_embeddings: Tensor of shape [T, C_Q]
        """
        if self.aggregation_method == 'mean':
            # Simple average pooling across queries
            frame_embeddings = frame_queries.mean(dim=1)  # [T, C_Q]
            
        elif self.aggregation_method == 'max':
            # Max pooling across queries
            frame_embeddings = frame_queries.max(dim=1)[0]  # [T, C_Q]
            
        elif self.aggregation_method == 'attention':
            # Attention-weighted aggregation
            # Compute attention weights for each query
            attn_weights = self.attention_weights(frame_queries)  # [T, N, 1]
            frame_embeddings = (frame_queries * attn_weights).sum(dim=1)  # [T, C_Q]
            
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return frame_embeddings
    
    def compute_relevance_scores(
        self,
        frame_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scalar relevance score s_t ∈ [0,1] for each frame.
    
        
        These scores measure the visual-text alignment between each frame 
        and the referring expression.
        
        Args:
            frame_embeddings: Tensor of shape [T, C_Q]
        
        Returns:
            scores: Tensor of shape [T], values in [0, 1]
        """
        # Apply MLP: σ(W₂·ReLU(W₁·e_t) + b)
        scores = self.score_mlp(frame_embeddings).squeeze(-1)  # [T]
        return scores
    
    def select_top_k_frames(
        self,
        scores: torch.Tensor,
        T_prime: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-K frames based on relevance scores while maintaining temporal order.
        
        After computing relevance scores {s_t}_{t=1}^T for all frames, we select 
        the top-T' frames with the highest scores, maintaining temporal order.
        
        For example, for the expression "scissors traveling," frames showing active 
        motion score >0.8, while idle frames score <0.3.
        
        Args:
            scores: Tensor of shape [T], relevance scores
            T_prime: Number of frames to select (if None, use top_k_ratio)
        
        Returns:
            selected_indices: Tensor of shape [T'], indices of selected frames (sorted)
            selected_scores: Tensor of shape [T'], scores of selected frames
        """
        T = scores.shape[0]
        
        # Determine number of frames to select
        if T_prime is None:
            T_prime = max(1, int(T * self.top_k_ratio))
        else:
            T_prime = min(T_prime, T)  # Ensure we don't select more than available
        
        # Get top-K indices based on scores
        _, top_indices = torch.topk(scores, k=T_prime, largest=True)
        
        # Sort indices to maintain temporal order
        selected_indices, _ = torch.sort(top_indices)
        selected_scores = scores[selected_indices]
        
        return selected_indices, selected_scores
    
    def forward(
        self,
        frame_queries: torch.Tensor,
        return_all_scores: bool = False,
        T_prime: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of key frame selector.
        
        Args:
            frame_queries: Tensor of shape [B, T, N_queries, C_Q] or [T, N_queries, C_Q]
                          B: batch size
                          T: number of frames
                          N_queries: number of object queries per frame
                          C_Q: query dimension
            return_all_scores: If True, return scores for all frames
            T_prime: Number of frames to select (if None, use top_k_ratio)
        
        Returns:
            selected_indices: Tensor of shape [B, T'] or [T']
            selected_scores: Tensor of shape [B, T'] or [T']
            all_scores: (optional) Tensor of shape [B, T] or [T] if return_all_scores=True
        """
        # Handle batch dimension
        if frame_queries.dim() == 4:
            # Batch processing: [B, T, N, C]
            B, T, N, C = frame_queries.shape
            
            # Process each batch separately
            selected_indices_list = []
            selected_scores_list = []
            all_scores_list = []
            
            for b in range(B):
                # Aggregate queries for this batch
                frame_emb = self.aggregate_frame_queries(frame_queries[b])  # [T, C_Q]
                
                # Compute relevance scores
                scores = self.compute_relevance_scores(frame_emb)  # [T]
                
                # Select top-K frames
                indices, sel_scores = self.select_top_k_frames(scores, T_prime)
                
                selected_indices_list.append(indices)
                selected_scores_list.append(sel_scores)
                if return_all_scores:
                    all_scores_list.append(scores)
            
            # Stack results across batch
            selected_indices = torch.stack(selected_indices_list, dim=0)  # [B, T']
            selected_scores = torch.stack(selected_scores_list, dim=0)  # [B, T']
            
            if return_all_scores:
                all_scores = torch.stack(all_scores_list, dim=0)  # [B, T]
                return selected_indices, selected_scores, all_scores
            
            return selected_indices, selected_scores, None
        
        elif frame_queries.dim() == 3:
            # Single sequence: [T, N, C]
            frame_embeddings = self.aggregate_frame_queries(frame_queries)  # [T, C_Q]
            scores = self.compute_relevance_scores(frame_embeddings)  # [T]
            selected_indices, selected_scores = self.select_top_k_frames(scores, T_prime)
            
            if return_all_scores:
                return selected_indices, selected_scores, scores
            
            return selected_indices, selected_scores, None
        
        else:
            raise ValueError(
                f"frame_queries must be 3D or 4D tensor, got shape {frame_queries.shape}"
            )


class KeyFrameSelectorWithLoss(KeyFrameSelector):
    """
    Extended version of KeyFrameSelector with optional supervision loss.
    
    If you have ground truth annotations indicating which frames are important
    (e.g., frames with object motion), you can add a supervision signal.
    
    Args:
        query_dim (int): Dimension of query features (C_Q)
        hidden_dim (int): Hidden dimension for MLP (d)
        top_k_ratio (float): Ratio of frames to select
        use_supervision (bool): Whether to use supervised training
        loss_weight (float): Weight for selection loss. Default: 0.1
    """
    
    def __init__(
        self,
        query_dim: int = 256,
        hidden_dim: int = 128,
        top_k_ratio: float = 0.5,
        aggregation_method: str = 'mean',
        use_supervision: bool = False,
        loss_weight: float = 0.1
    ):
        super().__init__(query_dim, hidden_dim, top_k_ratio, aggregation_method)
        self.use_supervision = use_supervision
        self.loss_weight = loss_weight
        
    def compute_selection_loss(
        self,
        predicted_scores: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for frame selection.
        
        This encourages the model to assign high scores to important frames
        (frames with object motion, appearance changes, etc.)
        
        Args:
            predicted_scores: [B, T] or [T], relevance scores
            target_masks: [B, T] or [T], binary labels (1 if frame should be selected)
        
        Returns:
            loss: scalar tensor
        """
        loss = F.binary_cross_entropy(
            predicted_scores,
            target_masks.float(),
            reduction='mean'
        )
        return loss * self.loss_weight
    
    def forward_with_loss(
        self,
        frame_queries: torch.Tensor,
        target_frame_importance: Optional[torch.Tensor] = None,
        T_prime: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional loss computation.
        
        Args:
            frame_queries: [B, T, N, C] or [T, N, C]
            target_frame_importance: [B, T] or [T], optional ground truth importance
            T_prime: Number of frames to select
        
        Returns:
            selected_indices: Selected frame indices
            selected_scores: Scores of selected frames
            all_scores: Scores for all frames
            loss: Selection loss (if use_supervision=True and targets provided)
        """
        # Get frame selection results
        selected_indices, selected_scores, all_scores = self.forward(
            frame_queries,
            return_all_scores=True,
            T_prime=T_prime
        )
        
        # Compute loss if supervision is enabled and targets are provided
        loss = None
        if self.use_supervision and target_frame_importance is not None:
            loss = self.compute_selection_loss(all_scores, target_frame_importance)
        
        return selected_indices, selected_scores, all_scores, loss


def apply_frame_selection_to_queries(
    frame_queries: torch.Tensor,
    selected_indices: torch.Tensor
) -> torch.Tensor:
    """
    Utility function to filter frame queries based on selected indices.
    
    Args:
        frame_queries: [B, T, N, C], all frame queries
        selected_indices: [B, T'], indices of selected frames
    
    Returns:
        selected_queries: [B, T', N, C], filtered queries
    """
    B, T, N, C = frame_queries.shape
    T_prime = selected_indices.shape[1]
    
    # Gather selected frames
    selected_queries = []
    for b in range(B):
        indices = selected_indices[b]  # [T']
        selected_queries.append(frame_queries[b, indices])  # [T', N, C]
    
    selected_queries = torch.stack(selected_queries, dim=0)  # [B, T', N, C]
    return selected_queries


def expand_masks_to_all_frames(
    predicted_masks: torch.Tensor,
    selected_indices: torch.Tensor,
    total_frames: int,
    interpolation_method: str = 'nearest'
) -> torch.Tensor:
    """
    Expand predicted masks from selected frames to all frames.
    
    Used during inference to generate masks for all frames even though
    only key frames were processed.
    
    Args:
        predicted_masks: [B, T', H, W], masks for selected frames
        selected_indices: [B, T'], indices of selected frames
        total_frames: T, total number of frames
        interpolation_method: 'nearest' or 'linear'
    
    Returns:
        expanded_masks: [B, T, H, W], masks for all frames
    """
    B, T_prime, H, W = predicted_masks.shape
    device = predicted_masks.device
    
    expanded_masks = []
    
    for b in range(B):
        indices = selected_indices[b].cpu().numpy()  # [T']
        masks_b = predicted_masks[b]  # [T', H, W]
        
        # Create full mask tensor
        full_masks = torch.zeros(total_frames, H, W, device=device)
        
        # Place predicted masks at selected indices
        full_masks[indices] = masks_b
        
        # Interpolate for non-selected frames
        if interpolation_method == 'nearest':
            # Nearest neighbor interpolation
            for t in range(total_frames):
                if t not in indices:
                    # Find nearest selected frame
                    nearest_idx = min(indices, key=lambda x: abs(x - t))
                    mask_idx = list(indices).index(nearest_idx)
                    full_masks[t] = masks_b[mask_idx]
        
        elif interpolation_method == 'linear':
            # Linear interpolation (more sophisticated)
            # Fill gaps between selected frames
            indices_sorted = sorted(indices)
            for i in range(len(indices_sorted) - 1):
                start_t = indices_sorted[i]
                end_t = indices_sorted[i + 1]
                
                if end_t - start_t > 1:
                    # Interpolate between start and end
                    start_mask = masks_b[list(indices).index(start_t)]
                    end_mask = masks_b[list(indices).index(end_t)]
                    
                    for t in range(start_t + 1, end_t):
                        alpha = (t - start_t) / (end_t - start_t)
                        full_masks[t] = (1 - alpha) * start_mask + alpha * end_mask
        
        expanded_masks.append(full_masks)
    
    expanded_masks = torch.stack(expanded_masks, dim=0)  # [B, T, H, W]
    return expanded_masks


if __name__ == "__main__":
    # Test the KeyFrameSelector
    print("Testing KeyFrameSelector...")
    
    # Create selector
    selector = KeyFrameSelector(
        query_dim=256,
        hidden_dim=128,
        top_k_ratio=0.5
    )
    
    # Test with batch data
    B, T, N, C = 2, 20, 100, 256
    frame_queries = torch.randn(B, T, N, C)
    
    print(f"\nInput shape: {frame_queries.shape}")
    
    # Forward pass
    selected_indices, selected_scores, all_scores = selector(
        frame_queries,
        return_all_scores=True
    )
    
    print(f"Selected indices shape: {selected_indices.shape}")
    print(f"Selected scores shape: {selected_scores.shape}")
    print(f"All scores shape: {all_scores.shape}")
    
    print(f"\nBatch 0 - Selected frame indices: {selected_indices[0].tolist()}")
    print(f"Batch 0 - Selected frame scores: {selected_scores[0].tolist()}")
    
    # Test filtering
    selected_queries = apply_frame_selection_to_queries(frame_queries, selected_indices)
    print(f"\nFiltered queries shape: {selected_queries.shape}")
    
    # Test with supervision
    print("\n\nTesting KeyFrameSelectorWithLoss...")
    selector_with_loss = KeyFrameSelectorWithLoss(
        query_dim=256,
        hidden_dim=128,
        top_k_ratio=0.5,
        use_supervision=True
    )
    
    # Create fake targets (some frames are important)
    target_importance = torch.zeros(B, T)
    target_importance[:, 5:15] = 1.0  # Frames 5-14 are important
    
    selected_indices, selected_scores, all_scores, loss = selector_with_loss.forward_with_loss(
        frame_queries,
        target_frame_importance=target_importance
    )
    
    print(f"Selection loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")