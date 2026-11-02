# Frozen Panels

`resnet18_tile_panel_48.csv` is the frozen 48-design panel used by the ViT-B/16
Tile DSE. It preserves only candidate selection metadata and design names; the
old CNN energy/latency values in the former result directory are intentionally not
retained because they used the pre-v5 Macro energy model.

`vit_l16_bayesian_panel_96_filtered.csv` is the ordered ViT-L/16 v2 panel. A GP
trained on the 24 valid v1 evaluations selects a 96-design shortlist from the
138-design compute window. The dense-small-Macro rule leaves 54 shortlist
designs; 17 remaining non-dense designs are appended as replacements so the
runner can continue until 48 designs complete successfully.
