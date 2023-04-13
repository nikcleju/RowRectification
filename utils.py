import numpy as np

def ridge_following(I, start_point, max_stride=1, stop_thresh = 0.5):

    assert len(I.shape)==2
    total_cols = I.shape[1]

    start_row, start_col = start_point

    # Prepare output vector, start with -1
    ridge = -1 * np.ones(total_cols, dtype=int)
    ridge[start_col] = start_row

    # Follow ridge to the right of start_point
    max_value = I[start_row, start_col]
    for i in range(start_col+1, total_cols):
        
        # row = ridge[i]
        # col = i
        lower_search_limit = max(0, ridge[i-1] - max_stride)
        upper_search_limit = min(I.shape[0], ridge[i-1] + max_stride + 1)
        ridge[i] = lower_search_limit + np.argmax(I[lower_search_limit : upper_search_limit, i])

        # Update max value
        max_value = max(max_value, I[ridge[i],i])

        # Check termination
        if I[ridge[i],i] < stop_thresh * max_value:
            ridge[i] = -1  # undo last point
            break

    # Follow ridge to the left of start_point
    max_value = I[start_row, start_col]
    for i in range(start_col-1, -1, -1):
        # Find next point on ridge
        lower_search_limit = max(0, ridge[i+1] - max_stride)
        upper_search_limit = min(I.shape[0], ridge[i+1] + max_stride + 1)
        ridge[i] = lower_search_limit + np.argmax(I[lower_search_limit : upper_search_limit, i])

        # Update max value
        max_value = max(max_value, I[ridge[i],i])

        # Check termination
        if I[ridge[i],i] < stop_thresh * max_value:
            ridge[i] = -1
            break

    return ridge

