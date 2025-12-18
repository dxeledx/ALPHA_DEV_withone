
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4
from sklearn.model_selection import train_test_split
import os


def _resolve_sessions(split_dict):
    """Return (session_T, session_E) regardless of underlying key names."""
    key_map = {k.lower(): k for k in split_dict.keys()}
    train_key = (key_map.get("session_t") or key_map.get("session_train") or
                 key_map.get("0train") or key_map.get("train"))
    test_key = (key_map.get("session_e") or key_map.get("session_test") or
                key_map.get("1test") or key_map.get("test"))
    session_keys = sorted(split_dict.keys())
    if train_key is None and session_keys:
        train_key = session_keys[0]
    if test_key is None and len(session_keys) > 1:
        test_key = session_keys[1]
    if train_key is None or test_key is None:
        raise KeyError("Unable to resolve session splits from dataset; available keys="
                       f"{list(split_dict.keys())}")
    return split_dict[train_key], split_dict[test_key]


def _concat_windows_dataset(concat_dataset):
    X_list = []
    y_list = []
    for windows_ds in concat_dataset.datasets:
        for idx in range(len(windows_ds)):
            window, target, _ = windows_ds[idx]
            X_list.append(window)
            y_list.append(target)
    X = np.stack(X_list).astype("float32")
    y = np.array(y_list).astype(int)
    return X, y


class BCICIV2a(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4 
    
    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("session")
        train_dataset, test_dataset = _resolve_sessions(splitted_ds)

        # load the data
        X, y = _concat_windows_dataset(train_dataset)
        X_test, y_test = _concat_windows_dataset(test_dataset)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)                                                                
        # self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, 
                                                                #  preprocessing_dict=self.preprocessing_dict, mode="train")
        # self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test, 
                                                                #  preprocessing_dict=self.preprocessing_dict, mode="test")


class BCICIV2aTVT(BaseDataModule):
    val_dataset = None
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4 

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        # Split by session
        splitted_ds = self.dataset.split("session")
        session1, session2 = _resolve_sessions(splitted_ds)
        
        # Load session 1 data
        X, y = _concat_windows_dataset(session1)

        # Split session 1: 80% train, 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.preprocessing_dict.get("seed", 42), stratify=y)

        # Load session 2 as test set
        X_test, y_test = _concat_windows_dataset(session2)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        # Create datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        # self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="train")
        # self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val, y_val, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="val")
        # self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test, y_test, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="test")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                        #   persistent_workers=True,          # ↩︎ keeps workers alive between epochs
                        #   prefetch_factor=4                 # ↩︎ each worker preloads 4 future batches                          
                        )


class BCICIV2aLOSO(BCICIV2a):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=self.all_subject_ids, dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        train_datasets = []
        val_datasets = []
        for subj_id in train_subjects:
            sess_T, sess_E = _resolve_sessions(splitted_ds[str(subj_id)].split("session"))
            train_datasets.append(sess_T)
            val_datasets.append(sess_E)
        _, test_dataset = _resolve_sessions(splitted_ds[str(self.subject_id)].split("session"))

        # load the data
        X_list, y_list = zip(*[_concat_windows_dataset(td) for td in train_datasets])
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        Xv_list, yv_list = zip(*[_concat_windows_dataset(td) for td in val_datasets])
        X_val = np.concatenate(Xv_list, axis=0)
        y_val = np.concatenate(yv_list, axis=0)
        X_test, y_test = _concat_windows_dataset(test_dataset)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

        # self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="train")
        # self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val, y_val, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="val")
        # self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test, y_test, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="test")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                        #   persistent_workers=True,          # ↩︎ keeps workers alive between epochs
                        #   prefetch_factor=4                 # ↩︎ each worker preloads 4 future batches                          
                        )
