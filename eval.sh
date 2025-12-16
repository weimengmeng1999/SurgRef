# python train_net.py \
#   --config-file configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml \
#   --eval-only MODEL.WEIGHTS output_sem_2/model_0099999.pth
#----------------------------------------------------------------------------------------------------------
#visualization
# python demo/demo.py --config-file configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml \
#   --input /nfs/home/mwei/EndoVis2018/mskfm/train/images/seq_1_frame019.jpg /nfs/home/mwei/EndoVis2018/mskfm/train/images/seq_1_frame136.jpg \
#   --output output_sem/vis \
#   --opts MODEL.WEIGHTS output_sem/model_0014999.pth

# python demo/demo.py --config-file configs/ade20k/instance-segmentation/maskformer2_R50_bs16_160k.yaml \
#   --input /nfs/home/mwei/mmsegmentation/data/robo/images/testing/1_p_3_80165_img.png /nfs/home/mwei/mmsegmentation/data/robo/images/testing/3_s_9_45000_img.png \
#   --output output_ins/vis_ins_2 \
#   --opts MODEL.WEIGHTS output_ins/model_final.pth
# python demo_agg/demo.py --config-file configs/ade20k/instance-segmentation/maskformer2_R50_bs16_160k.yaml \
#   --input /nfs/home/mwei/endovis2018-new/val/images\
#   --output  output_ins_endovis2018_2/endo18_agg\
#   --opts MODEL.WEIGHTS output_ins_endovis2018_2/model_final.pth
# python demo_inspart/demo.py --config-file configs/ade20k/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml \
#   --input /nfs/home/mwei/EndoVis2018-refine/train/images\
#   --output  /nfs/home/mwei/Mask2Former_inspart_output/output_inp_5/endo18_train_pan\
#   --opts MODEL.WEIGHTS /nfs/home/mwei/Mask2Former_inspart_output/output_inp_5/model_final.pth
# python train_net.py \
#   --config-file configs/ade20k/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml \
#   --num-gpus 1 \
#   --eval-only \
#   MODEL.WEIGHTS /nfs/home/mwei/Mask2Former_inspart_output/output_inp_5/model_final.pth
# python demo_inspart/demo.py --config-file configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k_inf.yaml \
#   --input /nfs/home/mwei/EndoVis2018-refine/val/images\
#   --output  /nfs/home/mwei/GuidedDistillation_output/output_inp_inf/endo18\
#   --opts MODEL.WEIGHTS /nfs/home/mwei/GuidedDistillation_output/output_inp_3/model_final.pth SSL.PERCENTAGE 100 SSL.TRAIN_SSL False
# python train_net.py \
#   --config-file configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml \
#   --num-gpus 1 \
#   --eval-only \
#   MODEL.WEIGHTS /nfs/home/mwei/GuidedDistillation_output/output_inp_3/model_final.pth\
#   SSL.PERCENTAGE 100 \
#   SSL.TRAIN_SSL False \
#   SSL.EVAL_WHO TEACHER
python train_net.py \
  --config-file configs/endovis/instance-part-segmentation/maskformer2_R50_bs16_160k.yaml \
  --num-gpus 1 \
  --eval-only \
  MODEL.WEIGHTS /nfs/home/mwei/GuidedDistillation_output/output_inp_13/model_0027999.pth\
  SSL.PERCENTAGE 100 \
  SSL.TRAIN_SSL False \
  SSL.EVAL_WHO TEACHER