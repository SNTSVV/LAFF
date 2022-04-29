# coding: utf-8
from __future__ import division
import json
import sys
from tools.utils import Utilities, MyLogger

params = {}


def parse_param(args):
    """
    parse the argument
    Parameter:
        args: the argument input from the command
    """
    options = None
    for index in range(1, len(args)):
        arg = args[index]
        arg_str = arg.strip()
        if arg_str[:1] == '-':
            if len(arg_str) < 2:
                print("Error at argument " + arg_str)
                return
            options = []
            params[arg_str[1:]] = options
        elif options != None:
            options.append(arg_str)
        else:
            print("Illegal parameter usage")
            return


def map_param_to_config(config, key, val):
    """
    map the parameter to the item in the configuration file
    set the value in the configuration file based on the parameter value
    Parameter:
        config: the configuration
        key: the key of an argument
        val: the value of an argument
    """
    if key == "run_alg":
        config["eval"]["run_alg"] = val[0] == "true"
    elif key == "run_eval":
        config["eval"]["run_eval"] = val[0] == "true"
    elif key == "alg":
        config["predict"]["algorithm"] = val[0]
    elif key == "type":
        config["predict"]["fill_type"] = val[0]
    elif key == "order":
        config["predict"]["fill_order"] = val[0]
    elif key == "round":
        config["predict"]["rounds"] = int(val[0])
    elif key == "use_filter":
        config["laff_param"]["use_filter"] = val[0] == "true"
    elif key == "use_local":
        config["laff_param"]["use_local"] = val[0] == "true"
    elif key == "eval_type":
        config["eval"]["fill_type"] = val
    elif key == "eval_order":
        config["eval"]["fill_orders"] = val
    elif key == "eval_algs":
        config["eval"]["algorithms"] = val


# configuration
config_file = Utilities.get_config_file_path()
with open(config_file)as cf:
    config = json.load(cf)

print("parse parameters")
parse_param(sys.argv)
print(params)

for key in params:
    map_param_to_config(config, key, params[key])

print("save parameters to the configuration file")
config_new = json.dumps(config, indent=2)
with open(config_file, 'w') as cf:
    cf.write(config_new)
    cf.close()
