###########################################################################
# Modified from: NTU
# Author: Meng Wei
###########################################################################

from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from .modeling.vita_criterion import VitaSetCriterion
from .modeling.vita_matcher import VitaHungarianMatcher
from .modeling.transformer_decoder.SurgRef import SURGREF
from .modeling.keyframe_selector import KeyFrameSelector, apply_frame_selection_to_queries
from transformers import BertModel, RobertaModel


@META_ARCH_REGISTRY.register()
class SurgRef(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        test_topk_per_image: int,
        # vita
        vita_module: nn.Module,
        vita_criterion: nn.Module,
        num_frames: int,
        num_classes: int,
        is_multi_cls: bool,
        apply_cls_thres: float,
        freeze_detector: bool,
        test_run_chunk_size: int,
        test_interpolate_chunk_size: int,
        is_coco: bool,
        output_threshold: float,
        lang_backbone: nn.Module,
        feature_resizer: nn.Module,
        freeze_text_encoder: bool,
        use_keyframe_selection: bool,
        keyframe_selector: nn.Module
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.test_topk_per_image = test_topk_per_image

        # vita hyper-parameters
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.vita_module = vita_module
        self.vita_criterion = vita_criterion
        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres

        if freeze_detector:
            for name, p in self.named_parameters():
                if not "vita_module" in name:
                    p.requires_grad_(False)
        self.test_run_chunk_size = test_run_chunk_size
        self.test_interpolate_chunk_size = test_interpolate_chunk_size

        self.is_coco = is_coco

        self.output_threshold = output_threshold
        self.resizer = feature_resizer
        self.text_encoder = lang_backbone
        print('whether freeze text encoder {}'.format(freeze_text_encoder))
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        self.use_keyframe_selection = use_keyframe_selection
        self.keyframe_selector = keyframe_selector


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        vita_deep_supervision = cfg.MODEL.VITA.DEEP_SUPERVISION

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        sim_weight = cfg.MODEL.VITA.SIM_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            vita_last_layer_num=cfg.MODEL.VITA.LAST_LAYER_NUM,
        )
        # Keyframe selector
        use_keyframe_selection = cfg.MODEL.KEYFRAME_SELECTOR.ENABLED
        if use_keyframe_selection:
            keyframe_selector = KeyFrameSelector(
                query_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM, 
                hidden_dim=cfg.MODEL.KEYFRAME_SELECTOR.HIDDEN_DIM, 
                top_k_ratio=cfg.MODEL.KEYFRAME_SELECTOR.TOP_K_RATIO  
            )
        if use_keyframe_selection:
            weight_dict["loss_keyframe"] = 0.5

        # Vita
        num_classes = sem_seg_head.num_classes
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        vita_module = SURGREF(cfg=cfg, in_channels=hidden_dim, aux_loss=vita_deep_supervision)

        # building criterion for vita inference
        vita_matcher = VitaHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        vita_weight_dict = {
            "loss_vita_ce": class_weight, "loss_vita_mask": mask_weight, "loss_vita_dice": dice_weight
        }
        if sim_weight > 0.0:
            vita_weight_dict["loss_vita_sim"] = sim_weight

        if vita_deep_supervision:
            vita_dec_layers = cfg.MODEL.VITA.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(vita_dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in vita_weight_dict.items()})
            vita_weight_dict.update(aux_weight_dict)
        vita_losses = ["vita_labels", "vita_masks"]
        if sim_weight > 0.0:
            vita_losses.append("fg_sim")

        vita_criterion = VitaSetCriterion(
            num_classes, 
            matcher=vita_matcher, 
            weight_dict=vita_weight_dict,
            eos_coef=cfg.MODEL.VITA.NO_OBJECT_WEIGHT,
            losses=vita_losses, 
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            sim_use_clip=cfg.MODEL.VITA.SIM_USE_CLIP,
        )
        # text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_encoder = RobertaModel.from_pretrained('roberta-base')
        resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.VITA.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # vita
            "vita_module": vita_module,
            "vita_criterion": vita_criterion,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_classes": num_classes,
            "is_multi_cls": cfg.MODEL.VITA.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.VITA.APPLY_CLS_THRES,
            "freeze_detector": cfg.MODEL.VITA.FREEZE_DETECTOR,
            "test_run_chunk_size": cfg.MODEL.VITA.TEST_RUN_CHUNK_SIZE,
            "test_interpolate_chunk_size": cfg.MODEL.VITA.TEST_INTERPOLATE_CHUNK_SIZE,
            "is_coco": cfg.DATASETS.TEST[0].startswith("coco"),
            "output_threshold": cfg.MODEL.VITA.TEST_OUTPUT_THRESHOLD,
            "lang_backbone": text_encoder,
            "feature_resizer": resizer,
            "freeze_text_encoder": cfg.MODEL.VITA.FREEZE_TEXT_ENCODER,
            "use_keyframe_selection": cfg.MODEL.KEYFRAME_SELECTOR.ENABLED,
            "keyframe_selector": keyframe_selector,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            return self.train_model(batched_inputs)
        else:
            # NOTE consider only B=1 case.
            return self.inference(batched_inputs[0])

    def train_model(self, batched_inputs):
        images = []

        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        lang_emb = [x['lang_tokens'].to(self.device) for x in batched_inputs]
        lang_emb = torch.cat(lang_emb, dim=0)

        lang_mask = [x['lang_mask'].to(self.device) for x in batched_inputs]
        lang_mask = torch.cat(lang_mask, dim=0)
        # lang_emb = torch.stack([x['lang_tokens'].to(self.device) for x in batched_inputs])  # (B, 40)
        # lang_mask = torch.stack([x['lang_mask'].to(self.device) for x in batched_inputs])  # (B, 40)

        lang_feat_all = self.text_encoder(lang_emb, attention_mask=lang_mask) # B, Nl, 768
        lang_feat_sentence = lang_feat_all.last_hidden_state
        lang_feat = lang_feat_all.pooler_output
        lang_feat = self.resizer(lang_feat)
        lang_mask = lang_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        lang_feat_sentence = lang_feat_sentence.permute(0, 2, 1)
        # print(f"Fixed lang_feat_sentence 1 shape: {lang_feat_sentence.shape}")
        # print(f"Fixed lang_mask 1 shape: {lang_mask.shape}")
        # print(f"Fixed images 1 shape: {images.tensor.shape}")
        bs = images.tensor.shape[0]
        lang_feat_sentence = lang_feat_sentence.repeat(bs, 1, 1)
        lang_mask = lang_mask.repeat(bs, 1, 1)
        # bs = images.tensor.shape[0]
        # B_video = lang_feat_sentence.shape[0]
        # repeat_factor = bs // B_video
        # lang_feat_sentence = lang_feat_sentence.repeat_interleave(repeat_factor, dim=0)
        # lang_mask = lang_mask.repeat_interleave(repeat_factor, dim=0)
        # Ensure lang_feat_sentence batch dimension matches images.tensor
        # if lang_feat_sentence.shape[0] != images.tensor.shape[0]:
        #     factor = images.tensor.shape[0] // lang_feat_sentence.shape[0]
        #     lang_feat_sentence = lang_feat_sentence.repeat(factor, 1, 1)  # ✅ Correctly expand batch dimension

        # if lang_mask.shape[0] != images.tensor.shape[0]:
        #     if lang_mask.dim() == 2:  # (B, 40)
        #         lang_mask = lang_mask.repeat(factor, 1)  # ✅ Only expand batch dimension
        #     elif lang_mask.dim() == 3:  # (B, 40, 1)
        #         lang_mask = lang_mask.repeat(factor, 1, 1)  # ✅ Only expand batch dimension

        # print(f"Fixed lang_feat_sentence shape: {lang_feat_sentence.shape}")
        # print(f"Fixed lang_mask shape: {lang_mask.shape}")
        # print(f"Fixed images shape: {images.tensor.shape}")

        features = self.backbone(images.tensor, lang_feat_sentence, lang_mask)

        BT = len(images) # batch * Frames
        T = self.num_frames if self.training else BT 
        B = BT // T

        outputs, frame_queries, mask_features = self.sem_seg_head(features, lang_feat, lang_mask)
        # print(f"\n=== BATCH SHAPE DEBUG ===")
        # print(f"T (frames per video) = {T}")
        # print(f"B (calculated batch size) = {B}")
        # print(f"frame_queries.shape = {frame_queries.shape}")
        # frame_queries: [3, 12, 100, 256] ; mask_features: [12, 256, 128, 144]
        # mask_features = self.vita_module.vita_mask_features(mask_features)
        # mask_features = mask_features.view(B, self.num_frames, *mask_features.shape[-3:])
        if self.use_keyframe_selection and self.training:
            # Key Frame Selection
            selected_indices, selected_scores, all_scores = self.keyframe_selector(
                frame_queries, return_all_scores=True
            )
            # print(f"\n{'='*60}")
            # print(f"{'='*60}")
            # print(f"Input: B={B}, T={T}")
            # print(f"Scores: mean={all_scores.mean():.3f}, std={all_scores.std():.3f}, min={all_scores.min():.3f}, max={all_scores.max():.3f}")
            # print(f"Selected indices shape: {selected_indices.shape}")
            # print(f"Selected indices per batch:")
            # for b in range(min(B, 3)):  # Show first 3 batches
            #     print(f"  Batch {b}: {selected_indices[b].tolist()}")
            
            T_selected = selected_indices.shape[1]
            # Filter frame_queries
            queries_for_vita = apply_frame_selection_to_queries(
                frame_queries, selected_indices
            )
            T_active = T_selected
            #Filter outputs from sem_seg_head

            outputs_for_criterion = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == B * T:
                    # Reshape to [B, T, ...]
                    v_reshaped = v.view(B, T, *v.shape[1:])
                    
                    # Select frames
                    v_selected = []
                    for b in range(B):
                        indices = selected_indices[b]
                        v_selected.append(v_reshaped[b, indices])
                    v_selected = torch.stack(v_selected, dim=0)
                    
                    # Flatten back to [B*T_selected, ...]
                    outputs_for_criterion[k] = v_selected.view(B * T_selected, *v_selected.shape[2:])
                else:
                    outputs_for_criterion[k] = v
            
            # Handle aux_outputs if they exist
            if 'aux_outputs' in outputs:
                outputs_for_criterion['aux_outputs'] = []
                for aux_out in outputs['aux_outputs']:
                    aux_filtered = {}
                    for k, v in aux_out.items():
                        if isinstance(v, torch.Tensor) and v.shape[0] == B * T:
                            v_reshaped = v.view(B, T, *v.shape[1:])
                            v_selected = []
                            for b in range(B):
                                indices = selected_indices[b]
                                v_selected.append(v_reshaped[b, indices])
                            v_selected = torch.stack(v_selected, dim=0)
                            aux_filtered[k] = v_selected.view(B * T_selected, *v_selected.shape[2:])
                        else:
                            aux_filtered[k] = v
                    outputs_for_criterion['aux_outputs'].append(aux_filtered)
            # Filter mask_features
            mask_features_reshaped = mask_features.view(B, T, *mask_features.shape[-3:])
            selected_mask_features = []
            for b in range(B):
                indices = selected_indices[b]
                selected_mask_features.append(mask_features_reshaped[b, indices])
            mask_features_selected = torch.stack(selected_mask_features, dim=0)
            mask_features = mask_features_selected.view(B * T_selected, *mask_features_selected.shape[-3:])
            
        else:
            print("No keyframe selection/ evaluation")
            queries_for_vita = frame_queries
            T_active = T
            outputs_for_criterion = outputs
            
        
        mask_features = self.vita_module.vita_mask_features(mask_features)
        # mask_features: [B*T_active, C, H, W]
        
        # NOW reshape to [B, T_active, C, H, W]
        mask_features_for_vita = mask_features.view(B, T_active, *mask_features.shape[-3:])

        # mask classification target
        frame_targets, clip_targets = self.prepare_targets(batched_inputs, images)
        if self.use_keyframe_selection and self.training:
            # frame_targets is a list of length B*T
            # Reshape to [B][T] structure
            frame_targets_reshaped = []
            for b in range(B):
                frame_targets_reshaped.append(frame_targets[b*T:(b+1)*T])
            
            # Select frames based on selected_indices
            frame_targets_selected = []
            for b in range(B):
                indices = selected_indices[b].cpu().tolist()  # [T'] - convert to list
                # Select only the frame targets at selected indices
                for idx in indices:
                    frame_targets_selected.append(frame_targets_reshaped[b][idx])
            
            frame_targets_for_loss = frame_targets_selected
            # clip_targets is a list of length B
            # Each has masks: [N_objects, T, H, W]
            clip_targets_selected = []
            for b in range(B):
                indices = selected_indices[b]  # [T'] - keep as tensor for indexing
                clip_target_b = clip_targets[b].copy()  # Make a copy
                
                selected_clip_target = {}
                for k, v in clip_target_b.items():
                    if k == 'masks' and torch.is_tensor(v):
                        # masks: [N_objects, T, H, W] -> select T dimension
                        selected_clip_target[k] = v[:, indices, :, :]  # [N_objects, T', H, W]
                    elif k == 'ids' and torch.is_tensor(v):
                        # ids: [N_objects, T] -> select T dimension
                        if v.dim() == 2 and v.shape[1] == T:
                            selected_clip_target[k] = v[:, indices]  # [N_objects, T']
                        else:
                            selected_clip_target[k] = v
                    elif k == 'frame_idx' and isinstance(v, list):
                        # frame_idx is a list of frame indices, select from it
                        selected_clip_target[k] = [v[i] for i in indices.cpu().tolist()]
                    else:
                        # Keep everything else as is (labels, video_len, etc.)
                        selected_clip_target[k] = v
                
                clip_targets_selected.append(selected_clip_target)
            
            clip_targets_for_loss = clip_targets_selected

            keyframe_selector_loss = (1.0 - selected_scores).mean()
            
        else:
            print("No keyframe selection/ evaluation")
            frame_targets_for_loss = frame_targets
            clip_targets_for_loss = clip_targets
        # bipartite matching-based loss
        losses, fg_indices = self.criterion(outputs_for_criterion, frame_targets_for_loss)

        vita_outputs = self.vita_module(queries_for_vita, lang_feat, lang_mask)
        vita_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", vita_outputs["pred_mask_embed"], mask_features_for_vita)
        for out in vita_outputs["aux_outputs"]:
            out["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", out["pred_mask_embed"], mask_features_for_vita)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        if keyframe_selector_loss is not None:
            losses['loss_keyframe'] = keyframe_selector_loss * 0.01
        # print(f"\nDEBUG BEFORE VITA CRITERION:")
        # print(f"vita_outputs['pred_masks'] shape: {vita_outputs['pred_masks'].shape}")
        # print(f"mask_features shape: {mask_features.shape}")
        # print(f"T_active: {T_active}, Original T: {T}")
        # print(f"\nframe_targets_for_vita structure:")
        # for b, target in enumerate(frame_targets):
        #     print(f"  Batch {b}:")
        #     for k, v in target.items():
        #         if isinstance(v, torch.Tensor):
        #             print(f"    {k}: shape {v.shape}, dtype {v.dtype}")
        #         else:
        #             print(f"    {k}: {type(v)}")
        # print(f"\nclip_targets_for_vita structure:")
        # for b, target in enumerate(clip_targets):
        #     print(f"  Batch {b}:")
        #     for k, v in target.items():
        #         if isinstance(v, torch.Tensor):
        #             print(f"    {k}: shape {v.shape}, dtype {v.dtype}")
        #         else:
        #             print(f"    {k}: {type(v)}")
        vita_loss_dict = self.vita_criterion(vita_outputs, clip_targets_for_loss, frame_targets_for_loss, fg_indices)
        vita_weight_dict = self.vita_criterion.weight_dict

        for k in vita_loss_dict.keys():
            if k in vita_weight_dict:
                vita_loss_dict[k] *= vita_weight_dict[k]
        losses.update(vita_loss_dict)
        return losses

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        frame_gt_instances = []
        clip_gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_classes_per_video = targets_per_video["instances"][0].gt_classes.to(self.device)
            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                _update_cls = gt_classes_per_video == -1
                gt_classes_per_video[_update_cls] = targets_per_frame.gt_classes[_update_cls]
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else: #polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
            gt_ids_per_video[gt_masks_per_video.sum(dim=(2,3)) == 0] = -1
            valid_bool_frame = (gt_ids_per_video != -1)
            valid_bool_clip = valid_bool_frame.any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_bool_clip].long() # N,
            gt_ids_per_video = gt_ids_per_video[valid_bool_clip].long()         # N, num_frames
            gt_masks_per_video = gt_masks_per_video[valid_bool_clip].float()    # N, num_frames, H, W
            valid_bool_frame = valid_bool_frame[valid_bool_clip]

            if len(gt_ids_per_video) > 0:
                min_id = max(gt_ids_per_video[valid_bool_frame].min(), 0)
                gt_ids_per_video[valid_bool_frame] -= min_id

            clip_gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video, "masks": gt_masks_per_video,
                    "video_len": targets_per_video["video_len"], "frame_idx": targets_per_video["frame_idx"],
                }
            )

            for f_i in range(self.num_frames):
                _cls = gt_classes_per_video.clone()
                _ids = gt_ids_per_video[:, f_i].clone()
                _mask = gt_masks_per_video[:, f_i].clone()

                valid = _ids != -1
                frame_gt_instances.append({
                    "labels": _cls[valid],
                    "ids": _ids[valid],
                    "masks": _mask[valid],
                })

        return frame_gt_instances, clip_gt_instances

    def inference(self, batched_inputs):
        frame_queries, mask_features = [], []
        num_frames = len(batched_inputs["image"])
        to_store = self.device if num_frames <= 36 else "cpu"

        lang_emb = batched_inputs['lang_tokens'].to(self.device)
        lang_mask = batched_inputs['lang_mask'].to(self.device)

        lang_feat_all = self.text_encoder(lang_emb, attention_mask=lang_mask) # B, Nl, 768
        lang_feat_sentence = lang_feat_all.last_hidden_state
        lang_feat_sentence = lang_feat_sentence.permute(0, 2, 1)
        lang_feat = lang_feat_all.pooler_output
        lang_feat = self.resizer(lang_feat)
        lang_mask = lang_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        for i in range(math.ceil(num_frames / self.test_run_chunk_size)):
            images = batched_inputs["image"][i*self.test_run_chunk_size : (i+1)*self.test_run_chunk_size]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            bs = images.tensor.shape[0]
            lang_feat_sentence_all = lang_feat_sentence.repeat(bs, 1, 1)
            lang_mask_all = lang_mask.repeat(bs, 1, 1)

            features = self.backbone(images.tensor, lang_feat_sentence_all, lang_mask_all)
            outputs, _frame_queries, _mask_features = self.sem_seg_head(features, lang_feat, lang_mask)

            _mask_features = self.vita_module.vita_mask_features(_mask_features)

            # BT is 1 as runs per frame
            frame_queries.append(_frame_queries[-1])    # T', fQ, C
            mask_features.append(_mask_features.to(to_store))  # T', C, H, W

        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation

        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs.get("width", image_size[1])

        if batched_inputs['dataset_name'] == 'mevis':
            merge = True
        else:
            merge = False
        del outputs, images, batched_inputs

        frame_queries = torch.cat(frame_queries)[None]  # 1, T, fQ, C
        mask_features = torch.cat(mask_features)        # T, C, H, W

        vita_outputs = self.vita_module(frame_queries, lang_feat, lang_mask)

        mask_cls = vita_outputs["pred_logits"][-1, 0]       # cQ, K+1
        mask_embed = vita_outputs["pred_mask_embed"][-1, 0] # cQ, C

        del vita_outputs

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]

        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        idx = scores.squeeze(-1) > self.output_threshold
        if not merge:
            top_score, idx = scores.squeeze(-1).topk(1, sorted=False)
        scores_per_video = scores.squeeze(-1)[idx]
        labels_per_video = labels[idx]
        mask_embed = mask_embed[idx]
        masks_per_video = []
        numerator = torch.zeros(len(mask_embed), dtype=torch.float, device=self.device)
        denominator = torch.zeros(len(mask_embed), dtype=torch.float, device=self.device)
        for i in range(math.ceil(len(mask_features) / self.test_interpolate_chunk_size)):
            m_f = mask_features[i*self.test_interpolate_chunk_size : (i+1)*self.test_interpolate_chunk_size].to(self.device)

            mask_pred = torch.einsum("qc,tchw->qthw", mask_embed, m_f)

            # upsample masks
            mask_pred = retry_if_cuda_oom(F.interpolate)(
                mask_pred,
                size=interim_size,
                mode="bilinear",
                align_corners=False,
            ) # cQ, T, H, W

            mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]

            interim_mask_soft = mask_pred.sigmoid()
            interim_mask_hard = interim_mask_soft > 0.5

            numerator += (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)
            denominator += interim_mask_hard.flatten(1).sum(1)

            mask_pred = F.interpolate(
                mask_pred, size=(out_height, out_width), mode="bilinear", align_corners=False
            ) > 0.
            mask_pred = mask_pred.sum(dim=0, keepdim=True).clamp(max=1)
            masks_per_video.append(mask_pred.to(to_store))
        masks_per_video = torch.cat(masks_per_video, dim=1)
        scores_per_video *= (numerator / (denominator + 1e-6))

        if self.is_coco:
            result = Instances((out_height, out_width))

            result.pred_masks = masks_per_video[:, 0].float()   # T=1 for COCO
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            result.scores = scores_per_video
            result.pred_classes = labels_per_video

            processed_results = [{"instances": result}]
        else:
            processed_results = {
                "image_size": (out_height, out_width),
                "pred_scores": scores_per_video.tolist(),
                "pred_labels": labels_per_video.tolist(),
                "pred_masks": masks_per_video.cpu(),
            }

        return processed_results


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output