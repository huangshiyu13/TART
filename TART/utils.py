"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: utils.py
"""
from __future__ import print_function

import os
import shutil
import numpy as np
import sys

def set_gpus():
    if len(sys.argv) < 2:
        print('please input GPU index')
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

def keep_xiaoshu(value,xiaoshuhou):
    tt = pow(10, xiaoshuhou)
    return int(value*tt)/float(tt)

def del_dir_under(dirname):
    os.system('rm -f {}/*'.format(dirname))

def check_file(filename):
    return os.path.isfile(filename)


def check_dir(dirname):
    return os.path.isdir(dirname)


def del_dir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

def del_create_dir(dirname):
    if check_dir(dirname):
        del_dir(dirname)
    create_dir(dirname)

def del_file(filename):
    if os.path.isfile(filename):
        os.system('rm ' + filename)

def create_dir(dirname):
    os.mkdir(dirname)

def new_dir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

    os.mkdir(dirname)

def del_dir_under(dirname):
    os.system('rm -f {}/*'.format(dirname))

def get_all_files(input_dir, suffix=None):
    files = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and (suffix is None or os.path.splitext(file_path)[1] == suffix):
            files.append(file_path)

    return files

def get_all_files_recursive(dirs, suffix=None):
    files = []
    for dir in dirs:
        for t in os.walk(dir):
            t_dir, t_cdirs, t_files = t
            if len(t_files) is not 0:
                for file in t_files:
                    file_path = os.path.join(t_dir, file)
                    if suffix is None or os.path.splitext(file_path)[1] == suffix:
                        print(file_path)
                        files.append(file_path)
    return files

def copy_file(from_file, to_file):
    os.system('cp \"{}\" \"{}\"'.format(from_file, to_file))


def move_file(from_file, to_file):
    os.system('mv \"{}\" \"{}\"'.format(from_file, to_file))

def get_filename(filepath):
    filepath = filepath.strip()
    while filepath and filepath[-1] == '/':
        filepath = filepath[:-1]

    file_s = filepath.split('/')
    if '.' in file_s[-1]:
        filename = file_s[-1].split('.')[0]
    else:
        filename = file_s[-1]
    return filename

def merge_dirs(dirs, output_dir):
    files = []

    for dir_now in dirs:
        files +=get_all_files(dir_now)

    for file in files:
        filename = get_filename(file)
        copy_file(file,output_dir+filename+file.split['.'][-1])


def convert_houzhui(data_dir, from_houzhui, to_houzhui):
    files = get_all_files(data_dir, from_houzhui)
    print('get {} {} files'.format(len(files), from_houzhui))

    for file in files:
        filename = get_filename(file)
        move_file(file, data_dir + filename + to_houzhui)

def convert_oldtonew(from_path, to_path):
    from_model = np.load(from_path, encoding='latin1').item()
    convert_dit = {}
    for key in from_model:
        convert_dit[key + ':0'] = from_model[key]
    np.save(to_path, convert_dit)

def convert_veryold2new(from_path,to_path):
    model_dict = np.load(from_path, encoding='latin1').item()
    model_dict_new = {}
    for key in model_dict:
        for i in range( len(model_dict[key])):
            if 'conv' in key:
                if i == 0:
                    model_dict_new['{}/filter:0'.format(key)] = model_dict[key][i]
                if i == 1:
                    model_dict_new['{}/biases:0'.format(key)] = model_dict[key][i]
            if 'fc' in key:
                if i == 0:
                    model_dict_new['{}/weights:0'.format(key)] = model_dict[key][i]
                if i == 1:
                    model_dict_new['{}/biases:0'.format(key)] = model_dict[key][i]

    np.save(to_path, model_dict_new)
