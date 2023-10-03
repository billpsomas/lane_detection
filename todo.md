    todo:
    2. currently the mask to evlauate (lanenet or hnet-fit) is commneted
    out, we need to have contant or config or something
    3. we need to improve results and fix the predict function
    1. consider using the original sized cooods of the lane points we want
    to transofrm back insted using the 128/64 coords from the image sise.
    maybe we need than to multiply the hnet inference's coeffs as they did
    in the reference
    2. use the transformed back lane points in the evaluation. maybe we need
    to divide the "get_points_after_hnet_and_fit_from_lanenet_cluster"
    mehtod because they are doing the same parsing from clustering lanenet
    in the evaluatation

- document new functions before merge to maste
- save when hnet is train the poly ordder and use it where ever needed (Hnet/train_hnet_v2.py:152)
