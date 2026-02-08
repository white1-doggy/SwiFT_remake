# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset, IterableDataset

# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nb
import nilearn
import random

from itertools import cycle
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

TASK_LABEL_RULES = {
    "WM": {
        "evs": [
            "0bk_faces", "0bk_places", "0bk_tools", "0bk_body",
            "2bk_faces", "2bk_places", "2bk_tools", "2bk_body",
        ],
        "label_map": {
            "0bk_faces": 0, "0bk_places": 0, "0bk_tools": 0, "0bk_body": 0,
            "2bk_faces": 1, "2bk_places": 1, "2bk_tools": 1, "2bk_body": 1,
        },
        "num_classes": 2,
    },
    "SOCIAL": {
        "evs": ["mental", "rnd"],
        "label_map": {"rnd": 0, "mental": 1},
        "num_classes": 2,
    },
    "EMOTION": {
        "evs": ["fear", "neut"],
        "label_map": {"neut": 0, "fear": 1},
        "num_classes": 2,
    },
    "MOTOR": {
        "evs": ["lh", "rh", "lf", "rf"],
        "label_map": {"lh": 0, "rh": 1, "lf": 0, "rf": 1},
        "num_classes": 2,
    },
    "LANGUAGE": {
        "evs": ["story", "math"],
        "label_map": {"math": 0, "story": 1},
        "num_classes": 2,
    },
    "RELATIONAL": {
        "evs": ["match", "relation"],
        "label_map": {"match": 0, "relation": 1},
        "num_classes": 2,
    },
    "GAMBLING": {
        "evs": ["loss", "win"],
        "label_map": {"loss": 0, "win": 1},
        "num_classes": 2,
    },
}

TASK_FIXED_TR = {
    "WM": 38,
    "SOCIAL": 32,
    "EMOTION": 25,
    "MOTOR": 17,
    "LANGUAGE": 22,
    "RELATIONAL": 22,
    "GAMBLING": 39,
}

TASK_NAME_LIST = [
    "WM",
    "SOCIAL",
    "EMOTION",
    "MOTOR",
    "LANGUAGE",
    "RELATIONAL",
    "GAMBLING",
]
TASK_NAME_TO_ID = {name: i for i, name in enumerate(TASK_NAME_LIST)}
NUM_TASKS = 7
GLOBAL_FIXED_TR = max(TASK_FIXED_TR.values())

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration),1)
        self.data = self._set_data(self.root, self.subject_dict)
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        if self.contrastive:
            num_frames = len(os.listdir(subject_path)) - 2
            y = []
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                y.append(y_loaded)
            y = torch.cat(y, dim=4)
            
            random_y = []
            
            full_range = np.arange(0, num_frames-sample_duration+1)
            # exclude overlapping sub-sequences within a subject
            exclude_range = np.arange(start_frame-sample_duration, start_frame+sample_duration)
            available_choices = np.setdiff1d(full_range, exclude_range)
            if len(available_choices) == 0:
                random_start_frame = start_frame
            else:
                random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
            load_fnames = [f'frame_{frame}.pt' for frame in range(random_start_frame, random_start_frame+sample_duration,self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                random_y.append(y_loaded)
            random_y = torch.cat(random_y, dim=4)
            return (y, random_y)

        else: # without contrastive learning
            y = []
            if self.shuffle_time_sequence: # shuffle whole sequences
                load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0,num_frames)),sample_duration//self.stride_within_seq)]
            else:
                load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_i = torch.load(img_path).unsqueeze(0)
                y.append(y_i)
            y = torch.cat(y, dim=4)
            return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")

class S1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        for i, subject in enumerate(subject_dict):
            sex,target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # target = self.label_dict[target] if isinstance(target, str) else target.float()

        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            rand_y = torch.nn.functional.pad(rand_y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 

class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, 'sub-'+subject_name)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        #age = self.label_dict[age] if isinstance(age, str) else age.float()
        
        #contrastive learning
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            rand_y = torch.nn.functional.pad(rand_y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

        # resting or task
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            if self.input_type == 'rest':
                # ABCD rest image shape: 79, 97, 85
                # latest version might be 96,96,95
                y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            elif self.input_type == 'task':
                # ABCD task image shape: 96, 96, 95
                # background value = 0
                # minmax scaled in brain (0~1)
                y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 
        

class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        # subject_list = [subj for subj in os.listdir(img_root) if subj.endswith('20227_2_0')] # only use release 2

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject20227 = str(subject_name)+'_20227_2_0'
            subject_path = os.path.join(img_root, subject20227)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        if self.contrastive:
                y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

                background_value = y.flatten()[0]
                y = y.permute(0,4,1,2,3)
                y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                y = y.permute(0,2,3,4,1)

                background_value = rand_y.flatten()[0]
                rand_y = rand_y.permute(0,4,1,2,3)
                rand_y = torch.nn.functional.pad(rand_y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                rand_y = rand_y.permute(0,2,3,4,1)

                return {
                    "fmri_sequence": (y, rand_y),
                    "subject_name": subject_name,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex
                }
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)
            return {
                        "fmri_sequence": y,
                        "subject_name": subject_name,
                        "target": target,
                        "TR": start_frame,
                        "sex": sex,
                    } 
    
class Dummy(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, total_samples=100000)
        

    def _set_data(self, root, subject_dict):
        data = []
        for k in range(0,self.total_samples):
            data.append((k, 'subj'+ str(k), 'path'+ str(k), self.stride))
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([val for val in range(len(data))]).reshape(-1, 1)
            
        return data

    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        _, subj, _, sequence_length = self.data[idx]
        y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16) #self.y[seq_idx]
        sex = torch.randint(0,2,(1,)).float()
        target = torch.randint(0,2,(1,)).float()

        if self.contrastive:
            rand_y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16)
            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subj,
                "target": target,
                "TR": 0,
                }
        else:
            return {
                    "fmri_sequence": y,
                    "subject_name": subj,
                    "target": target,
                    "TR": 0,
                    "sex": sex,
                    } 


class HCPTask(BaseDataset):
    def __init__(self, **kwargs):
        task_name = kwargs.get("downstream_task")
        if task_name not in TASK_LABEL_RULES:
            raise ValueError(f"Unsupported HCPTask downstream task: {task_name}")
        kwargs["sequence_length"] = TASK_FIXED_TR[task_name]
        kwargs["stride_within_seq"] = 1
        self.task_name = task_name
        self.task_config = TASK_LABEL_RULES[task_name]
        self.task_sequence_length = TASK_FIXED_TR[task_name]
        self.frame_cache = {}
        super().__init__(**kwargs)

    def _get_frame_files(self, split_data_path):
        if split_data_path in self.frame_cache:
            return self.frame_cache[split_data_path]
        pt_files = [
            fname
            for fname in os.listdir(split_data_path)
            if fname.endswith(".pt") and fname not in {"voxel_mean.pt", "voxel_std.pt"}
        ]

        def _extract_index(fname):
            stem = Path(fname).stem
            digits = "".join([c for c in stem if c.isdigit()])
            return int(digits) if digits else float("inf")

        pt_files = sorted(pt_files, key=lambda f: (_extract_index(f), f))
        self.frame_cache[split_data_path] = pt_files
        return pt_files

    def _load_evs_onsets(self, ev_file_path):
        if not os.path.exists(ev_file_path):
            return []
        try:
            evs = np.loadtxt(ev_file_path)
        except ValueError:
            return []
        if evs.size == 0:
            return []
        evs = np.atleast_2d(evs)
        onsets = evs[:, 0].tolist()
        return [int(round(onset)) for onset in onsets]

    def _set_data(self, root, subject_dict):
        data = []
        for i, subject in enumerate(subject_dict):
            task_root = os.path.join(root, subject, self.task_name)
            if not os.path.exists(task_root):
                continue
            for run in ["LR", "RL"]:
                split_data_path = os.path.join(task_root, run, "Split_data")
                evs_path = os.path.join(task_root, run, "EVs")
                if not os.path.exists(split_data_path) or not os.path.exists(evs_path):
                    continue

                frame_files = self._get_frame_files(split_data_path)
                num_frames = len(frame_files)
                if num_frames == 0:
                    continue

                for ev_name in self.task_config["evs"]:
                    ev_file = os.path.join(evs_path, f"{ev_name}.txt")
                    onsets = self._load_evs_onsets(ev_file)
                    for onset in onsets:
                        if onset + self.task_sequence_length > num_frames:
                            continue
                        label = self.task_config["label_map"][ev_name]
                        data.append(
                            (
                                i,
                                subject,
                                split_data_path,
                                onset,
                                num_frames,
                                label,
                                run,
                            )
                        )

        if self.train:
            self.target_values = np.array([tup[5] for tup in data]).reshape(-1, 1)
        return data

    def _load_task_sequence(self, split_data_path, start_frame, num_frames, sample_duration):
        frame_files = self._get_frame_files(split_data_path)
        end_frame = start_frame + sample_duration
        load_indices = range(start_frame, end_frame, self.stride_within_seq)
        y = []
        for frame_idx in load_indices:
            if frame_idx >= num_frames:
                break
            fname = frame_files[frame_idx]
            img_path = os.path.join(split_data_path, fname)
            y_i = torch.load(img_path).unsqueeze(0)
            y.append(y_i)

        if self.with_voxel_norm:
            for fname in ["voxel_mean.pt", "voxel_std.pt"]:
                norm_path = os.path.join(split_data_path, fname)
                if os.path.exists(norm_path):
                    y_norm = torch.load(norm_path).unsqueeze(0)
                    y.append(y_norm)

        return torch.cat(y, dim=4)

    def __getitem__(self, index):
        _, subject, split_data_path, start_frame, num_frames, target, run = self.data[index]

        if self.contrastive:
            y = self._load_task_sequence(
                split_data_path,
                start_frame,
                num_frames,
                self.task_sequence_length,
            )

            full_range = np.arange(0, num_frames - self.task_sequence_length + 1)
            exclude_range = np.arange(start_frame - self.task_sequence_length, start_frame + self.task_sequence_length)
            available_choices = np.setdiff1d(full_range, exclude_range)
            if len(available_choices) == 0:
                random_start_frame = start_frame
            else:
                random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
            rand_y = self._load_task_sequence(
                split_data_path,
                random_start_frame,
                num_frames,
                self.task_sequence_length,
            )

            background_value = y.flatten()[0]
            y = y.permute(0, 4, 1, 2, 3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value)
            y = y.permute(0, 2, 3, 4, 1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0, 4, 1, 2, 3)
            rand_y = torch.nn.functional.pad(rand_y, (8, 7, 2, 1, 11, 10), value=background_value)
            rand_y = rand_y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": 0,
                "run": run,
            }

        y = self._load_task_sequence(
            split_data_path,
            start_frame,
            num_frames,
            self.task_sequence_length,
        )

        background_value = y.flatten()[0]
        y = y.permute(0, 4, 1, 2, 3)
        y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value)
        y = y.permute(0, 2, 3, 4, 1)

        return {
            "fmri_sequence": y,
            "subject_name": subject,
            "target": target,
            "TR": start_frame,
            "sex": 0,
            "run": run,
        }
