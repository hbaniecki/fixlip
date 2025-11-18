# experiments

Scripts for running main experiments:

- `pointing_game_*.py` evaluate FIxLIP with a pointing game for CLIP/SigLIP on ImageNet-1k (see `data` directory)
- `explain_mscoco.py`, `explain_mscoco_siglip.py` computes FIxLIP for CLIP/SigLIP on MS COCO
- `insertion_deletion.py`, `insertion_deletion_siglip.py` evaluates the computed FIxLIP for CLIP/SigLIP on MS COCO
- `faithfulness.py` evaluates the computed FIxLIP (and other approaches, see `exclip` and `gradeclip` directories) for CLIP on MS COCO

Other `.py`/`.sh` files are for running experiments with Slurm.