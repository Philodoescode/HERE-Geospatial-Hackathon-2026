"""
DOGE-Adapted Bezier Centerline Extraction.

A standalone pipeline that adapts the DOGE (Differentiable Optimization of
Bezier Graph) framework for road centerline extraction from GPS traces
(VPD + HPD/Probe data) instead of satellite imagery.

Pipeline:
    1. data_loader    - Load VPD/HPD traces, clip, project to UTM
    2. rasterizer     - Rasterize GPS traces into a density segmentation mask
    3. tiling         - Split large areas into overlapping tiles
    4. bezier_graph   - Bezier Graph data structure (PyTorch tensors)
    5. diff_renderer  - Differentiable rendering (DiffVG or PyTorch fallback)
    6. diffalign      - Geometric optimization (5-term loss function)
    7. topoadapt      - Discrete topology refinement operators
    8. doge_optimizer  - Main optimization loop per tile
    9. run            - End-to-end entry point
"""
