import numpy as np

def ridge_following(I, start_point, max_stride=1, stop_thresh = 0.5):

    assert len(I.shape)==2
    cols = I.shape[1]

    start_row, start_col = start_point

    # Prepare output vector, start with -1
    ridge = -1 * np.ones(cols, dtype=int)
    ridge[start_col] = start_row

    # Follow ridge to the right of start_point
    max_value = I[start_row, start_col]
    for i in range(start_col+1, cols):
        
        ridge[i] = ridge[i-1] - max_stride + np.argmax(I[ridge[i-1] - max_stride : ridge[i-1] + max_stride + 1, i])

        # Update max value
        max_value = max(max_value, I[ridge[i],i])

        # Check termination
        if I[ridge[i],i] < stop_thresh * max_value:
            ridge[i] = -1
            break

    # Follow ridge to the left of start_point
    max_value = I[start_row, start_col]
    for i in range(start_col-1, -1, -1):
        # Find next point on ridge
        ridge[i] = ridge[i+1] - max_stride + np.argmax(I[ridge[i+1] - max_stride : ridge[i+1] + max_stride + 1, i])

        # Update max value
        max_value = max(max_value, I[ridge[i],i])

        # Check termination
        if I[ridge[i],i] < stop_thresh * max_value:
            ridge[i] = -1
            break

    return ridge
