# wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl
# python tools/process_ckpt.py
# python train_net_SurgRef.py \
#     --config-file configs/SurgRef_SWIN_bs8.yaml \
#     --num-gpus 4 --dist-url auto --eval-only \
#     OUTPUT_DIR outputs/endovis_rs17
# python train_net_SurgRef.py \
#     --config-file configs/SurgRef_SWIN_bs8.yaml \
#     --num-gpus 1 --dist-url auto \
#     MODEL.WEIGHTS /nfs/home/mwei/MeViS/model_final_86143f.pkl \
#     OUTPUT_DIR outputs/endovis_rs17_surgref_key_16_bs16
# python train_net_SurgRef.py \
#     --config-file configs/SurgRef_SWIN_bs8.yaml \
#     --num-gpus 1 --dist-url auto \
#     MODEL.WEIGHTS /nfs/home/mwei/MeViS/outputs/endovis_rs17/model_final.pth \
#     OUTPUT_DIR outputs/endovis_rs17_surgref_key_16_debug
# python train_net_SurgRef.py \
#     --config-file configs/SurgRef_SWIN_bs8.yaml \
#     --num-gpus 1 --dist-url auto --eval-only \
#     MODEL.WEIGHTS /nfs/home/mwei/MeViS/outputs/endovis_rs17/model_final.pth \
#     OUTPUT_DIR outputs/endovis_rs17_surgref_key_16_bs16
# python train_net_SurgRef.py \
#     --config-file configs/SurgRef_SWIN_bs8.yaml \
#     --num-gpus 1 --dist-url auto --eval-only \
#     MODEL.WEIGHTS /nfs/home/mwei/MeViS/outputs/endovis_rs17_surgref_key_16_debug/model_0059999.pth \
#     OUTPUT_DIR outputs/endovis_rs17_surgref_key_16_debug
python train_net_SurgRef.py \
    --config-file configs/SurgRef_SWIN_bs1_EIM18.yaml \
    --num-gpus 1 --dist-url auto \
    MODEL.WEIGHTS /nfs/home/mwei/MeViS/outputs/endovis_rs17/model_final.pth \
    OUTPUT_DIR outputs/endovis_rs18_surgref_key_16_debug