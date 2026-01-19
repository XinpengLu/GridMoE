import json
import os


def is_json_serializable(v):
    # copy from safepo
    try:
        json.dumps(v)
        return True
    except:
        return False


def convert_json(obj):
    # copy from safepo
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and "lambda" not in obj.__name__:
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)

def write2json(log_dir, config):
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4)
    with open(os.path.join(log_dir, "config.json"), "w") as out:
        out.write(output)


def task_nums2str(task_nums):
    return '-'.join(map(str, task_nums))