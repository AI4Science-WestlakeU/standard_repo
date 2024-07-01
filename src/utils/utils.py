import torch
import numpy as np
import yaml
import argparse
from datetime import datetime
import os
import sys
from termcolor import colored
import pdb

COLOR_LIST = ["b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine",
             "b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine"]

#basic utils
class Printer(object):
    ## Example, to print code running time between two p.print() calls
    # p.print(f"test_start", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))
def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string
p = Printer(n_digits=6)

# config utils
def add_args_from_config(parser):
    args=parser.parse_args()
    with open(args.config, 'r') as file:
        config=yaml.safe_load(file)

    existing_args = {action.dest for action in parser._actions}
    
    # Add arguments from the config file to the parser
    for key, value in config.items():
        # If the argument is not already added, add it to the parser
        if key not in existing_args:
            parser.add_argument(f'--{key}', type=type(value), default=value)
    
    return parser
def save_config_from_args(args):
    config_dict = {k: v for k, v in vars(args).items() if k != 'config'}
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    config_dir = args.exp_path+'/results/'+time_now
    os.makedirs(config_dir, exist_ok=True)  # Ensure the directory exists
    config_file_path = os.path.join(config_dir, 'config.yaml')
    with open(config_file_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)

    return 


# result analysis
def caculate_confidence_interval(data):
    ''''
    input example: abs(pred_design-pred_simu)
    '''
    list_dim=range(data.dim())
    if data.dim()>1:
        MAE_batch_size = torch.mean(data, dim=tuple(list_dim[1:]))
    else:
        MAE_batch_size =data
    mean = torch.mean(MAE_batch_size)
    
    std_dev = torch.std(MAE_batch_size)
    min_value=min(MAE_batch_size)
    confidence_level = 0.95
    # pdb.set_trace()
    n = len(data)
    # kk = stats.t.ppf((1 + confidence_level) / 2, n - 1) * (std_dev / (n ** 0.5))
    margin_of_error= std_dev* 1.96/ torch.sqrt(torch.tensor(n,dtype=float))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    print("mean:", mean.item())
    print("std:", std_dev.item())
    print(f"margin_of_error:", margin_of_error)

    return mean,std_dev,margin_of_error,min_value


#training utils
def caculate_num_parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    # pdb.set_trace()
    return
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return