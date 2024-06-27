import lmdb
import os
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import json
import io
PROTOCOL_LEVEL = 4


def read_lmdb(root_dir):
    db_env = lmdb.open(root_dir, readonly=True, lock=False)
    return db_env


def write_lmdb(root_dir, map_size=None):
    if map_size is not None:
        db_env = lmdb.open(root_dir, map_size=map_size)
    else:
        db_env = lmdb.open(
            root_dir,
        )
    return db_env


def read_folder(db_env, path=""):
    # 忽略结束的"/"
    if path.endswith("/"):
        path = path[: len(path) - 1]
    with db_env.begin() as txn:
        subs = txn.get("{}/__keys__".format(path).encode())
        if subs is None:
            return None
        subs = pickle.loads(subs)
    return subs


def read_txt(db_env, path):
    with db_env.begin() as txn:
        file_content = txn.get(path.encode())
        file_content = pickle.loads(file_content)
        file_content = file_content.decode("utf-8")
    return file_content


def read_json(db_env, path):
    with db_env.begin() as txn:
        file_content = txn.get(path.encode())
        file_content = pickle.loads(file_content)
        file_content = file_content.decode("utf-8")
        json_c = json.loads(file_content)
    return json_c


def read_image(db_env, path, grayscale=False):
    # return bgr image if not grayscale
    with db_env.begin() as txn:
        file_content = txn.get(path.encode())
        file_content = pickle.loads(file_content)
        image = cv2.imdecode(
            np.frombuffer(file_content, np.uint8),
            cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR,
        )
        if not grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_binary(db_env, path):
    with db_env.begin() as txn:
        file_content = txn.get(path.encode())
        file_content = pickle.loads(file_content)
    return file_content


def write_file(db_env, prefix, f_name, file_folder):
    f_key = f_name
    with open(os.path.join(file_folder, f_name), "rb") as f:
        f_content = f.read()
    try:
        with db_env.begin(write=True) as txn:
            txn.put(
                "{}/{}".format(prefix, f_key).encode(),
                pickle.dumps(f_content, protocol=4),
            )
    except lmdb.MapFullError:
        db_env.set_mapsize(db_env.info()["map_size"] + 1024 * 1024 * 1024)
        with db_env.begin(write=True) as txn:
            txn.put(
                "{}/{}".format(prefix, f_key).encode(),
                pickle.dumps(f_content, protocol=4),
            )


def write_keys(db_env, prefix, keys):
    try:
        with db_env.begin(write=True) as txn:
            txn.put(
                "{}/__keys__".format(prefix).encode(),
                pickle.dumps(keys, protocol=PROTOCOL_LEVEL),
            )
    except lmdb.MapFullError:
        db_env.set_mapsize(db_env.info()["map_size"] + 512 * 1024 * 1024)
        with db_env.begin(write=True) as txn:
            txn.put(
                "{}/__keys__".format(prefix).encode(),
                pickle.dumps(keys, protocol=PROTOCOL_LEVEL),
            )


def write_folder(db_env, folder, prefix="", folder_filter=None):
    sub_items = os.listdir(folder)
    if folder_filter is not None:
        sub_items = list(filter(folder_filter, sub_items))
    write_keys(db_env, prefix, sub_items)
    for sub_item in tqdm(sub_items):
        if os.path.isdir(os.path.join(folder, sub_item)):
            sub_prefix = "{}/{}".format(prefix, sub_item)
            write_folder(db_env, os.path.join(folder, sub_item), sub_prefix)
        else:
            write_file(db_env, prefix, sub_item, folder)


def write_sample(db_env, sample, index):
    try:
        with db_env.begin(write=True) as txn:
            txn.put(f"{index}".encode(), pickle.dumps(
                sample, protocol=PROTOCOL_LEVEL))
    except lmdb.MapFullError:
        db_env.set_mapsize(db_env.info()["map_size"] + 1024 * 1024 * 1024)
        with db_env.begin(write=True) as txn:
            txn.put(f"{index}".encode(), pickle.dumps(
                sample, protocol=PROTOCOL_LEVEL))


def write_file_content(db_env, f_key, f_content):
    try:
        with db_env.begin(write=True) as txn:
            txn.put(f_key.encode(), pickle.dumps(f_content, protocol=4))
    except lmdb.MapFullError:
        db_env.set_mapsize(db_env.info()["map_size"] + 1024 * 1024 * 1024)
        with db_env.begin(write=True) as txn:
            txn.put(f_key.encode(), pickle.dumps(f_content, protocol=4))


def is_dir(db_env, path):
    with db_env.begin(write=False) as txn:
        subs = txn.get("{}/__keys__".format(path).encode())
        if subs is None:
            return False
    return True

    
def copy_dir2(in_env, in_folder,out_env,out_folder):
    # copy folder keys
    in_folder_keys = read_folder(in_env, in_folder)
    exist_folder_keys = read_folder(out_env, out_folder)
    if exist_folder_keys is None:
        exist_folder_keys = []
    exist_folder_keys.extend(list(set(in_folder_keys)-set(exist_folder_keys)))
    write_keys(out_env, out_folder, exist_folder_keys)
    for f_key in in_folder_keys:
        f_path = f"{in_folder}/{f_key}"
        f_opath=f"{out_folder}/{f_key}"
        if is_dir(in_env, f_path):
            copy_dir2(in_env, f_path,out_env, f_opath)
        else:
            # copy content
            f_content = read_binary(in_env, f_path)
            write_file_content(out_env, f_path, f_content)


def write_numpy_compressed(db_env, f_key, np_data: np.ndarray):
    try:
        with db_env.begin(write=True) as txn:
            compressed_array = io.BytesIO()
            np.savez_compressed(compressed_array, np_data)
            compressed_array.seek(0)
            txn.put(f_key.encode(), compressed_array.read())
    except lmdb.MapFullError:
        db_env.set_mapsize(db_env.info()["map_size"] + 1024 * 1024 * 1024)
        with db_env.begin(write=True) as txn:
            compressed_array = io.BytesIO()
            np.savez_compressed(compressed_array, np_data)
            compressed_array.seek(0)
            txn.put(f_key.encode(), compressed_array.read())


def read_numpy_compressed(db_env, f_key):
    with db_env.begin(write=False) as txn:
        compressed_array = io.BytesIO(txn.get(f_key.encode()))
        feature = np.load(compressed_array)["arr_0"]
        return feature

def copy_dir(in_env, out_env, folder):
    copy_dir(in_env,folder,out_env,folder)

def rm_file_(db_env, path):
    #直接删除文件项
    with db_env.begin(write=True) as txn:
        txn.delete(path.encode())
        
def rm_in_parent_(db_env,path):
    #将这项从父目录中删除
    split_loc=path.rfind("/")
    parent_folder=path[:split_loc]
    parent_items=read_folder(db_env, parent_folder)
    if parent_items is not None:
        try:
            parent_items.remove(path[split_loc+1:])
            write_keys(db_env, parent_folder, parent_items)
        except ValueError:
            pass
        
def rm_file(db_env, path):
    rm_file_(db_env,path)
    rm_in_parent_(db_env,path)
            
def rm_dir(db_env,folder):
    sub_items=read_folder(db_env, folder)
    if sub_items is not None:
        with db_env.begin(write=True) as txn:
            txn.delete("{}/__keys__".format(folder).encode())
        #将这项从父目录中删除
        rm_in_parent_(db_env,folder)
    for sub_item in sub_items:
        sub_path=f"{folder}/{sub_item}"
        if is_dir(db_env, sub_path):
            rm_dir(db_env, sub_path)
        else:
            rm_file_(db_env, sub_path)