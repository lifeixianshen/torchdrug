separator = ">" * 30
line = "-" * 30

def time(seconds):
    """
    Format time as a string.

    Parameters:
        seconds (float): time in seconds
    """
    sec_per_min = 60
    sec_per_hour = 60 * 60
    sec_per_day = 24 * 60 * 60

    if seconds > sec_per_day:
        return "%.2f days" % (seconds / sec_per_day)
    elif seconds > sec_per_hour:
        return "%.2f hours" % (seconds / sec_per_hour)
    elif seconds > sec_per_min:
        return "%.2f mins" % (seconds / sec_per_min)
    else:
        return "%.2f secs" % seconds


def long_array(array, truncation=10, display=3):
    """
    Format an array as a string.

    Parameters:
        array (array_like): array-like data
        truncation (int, optional): truncate array if its length exceeds this threshold
        display (int, optional): number of elements to display at the beginning and the end in truncated mode
    """
    if len(array) <= truncation:
        return f"{array}"
    return f"{str(array[:display])[:-1]}, ..., {str(array[-display:])[1:]}"