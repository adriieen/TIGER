import os
import json
import numpy as np
from typing import Any, Tuple
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
# from pytorch_lightning.core.mixins import HyperparametersMixin
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import Dict, Iterable, List, Iterator
from rich import print
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_(message: str):
    print(message)


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class MP3DDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0, #seconds
        normalize_audio: bool = False,
        subset_percentile: float = None,
        subset_mode: str = "close",
        subset_fixed_samples: int = None,
        subset_seed: int = 42,
        subset_range: tuple = None,
        in_memory: bool = False,
    ) -> None:
        super().__init__()
        self.in_memory = in_memory
        self.cached_mix = None
        self.cached_sources = None
        self.EPS = 1e-8
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            
        self.n_src = n_src
        self.test = self.seg_len is None
        if subset_percentile is None and "TIGER_SUBSET_PERCENTILE" in os.environ:
            try:
                subset_percentile = float(os.environ["TIGER_SUBSET_PERCENTILE"])
            except ValueError:
                subset_percentile = None
        if subset_mode is None or subset_mode == "close":
            subset_mode = os.environ.get("TIGER_SUBSET_MODE", subset_mode)
        self.subset_percentile = subset_percentile
        self.subset_mode = subset_mode

        self.subset_fixed_samples = subset_fixed_samples
        self.subset_seed = subset_seed
        self.subset_range = subset_range
        self.subset_range = subset_range

        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []

        if self.n_src == 1: # speaker isolation
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print_(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = len(self.mix)

        elif self.n_src == 2: # 2 speaker separation
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]

            print_(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
                )
            )
            # --- Découpage du subset (déciles disjoints / ranges / percentiles cumulés) ---
            if not self.test and (self.subset_range is not None or self.subset_percentile is not None or self.subset_fixed_samples is not None):
                total_utts = len(mix_infos)

                if self.subset_range is not None:
                    # Décile disjoint ou tranche explicite [start_pct, end_pct]
                    start_pct, end_pct = self.subset_range
                    start_idx = int(total_utts * start_pct)
                    end_idx = int(total_utts * end_pct) if end_pct < 1.0 else total_utts
                    pool_indices = list(range(start_idx, end_idx))
                    print_(
                        f"[EchoSet] Tranche disjointe active : [{start_pct*100:.1f}% - {end_pct*100:.1f}%] "
                        f"-> indices {start_idx} à {end_idx} ({len(pool_indices)} paires)"
                    )
                elif self.subset_mode == "random":
                    # Groupe contrôle / Baseline aléatoire
                    pool_indices = list(range(total_utts))
                elif self.subset_percentile is not None and self.subset_percentile < 1.0:
                    n_pool = max(1, int(total_utts * self.subset_percentile))
                    if self.subset_mode == "close":
                        pool_indices = list(range(n_pool))
                    elif self.subset_mode == "far":
                        pool_indices = list(range(total_utts - n_pool, total_utts))
                else:
                    pool_indices = list(range(total_utts))

                # Étape 2 : Sous-échantillonnage uniforme à taille fixe N si demandé
                if self.subset_fixed_samples is not None and self.subset_fixed_samples < len(pool_indices):
                    rng = np.random.RandomState(self.subset_seed)
                    selected_indices = sorted(rng.choice(pool_indices, size=self.subset_fixed_samples, replace=False))
                else:
                    selected_indices = pool_indices

                # Application de la sélection
                mix_infos = [mix_infos[i] for i in selected_indices]
                sources_infos = [[src[i] for i in selected_indices] for src in sources_infos]

                print_(
                    f"[EchoSet] Subset '{self.subset_mode}' activé : "
                    f"{len(mix_infos)} paires sélectionnées sur un pool de {len(pool_indices)} "
                    f"(taille totale dataset: {total_utts})"
                )

            self.mix = mix_infos
            self.sources = sources_infos
            self.length = len(self.mix)

            if self.in_memory:
                print_(f"[EchoSet] Preloading {self.length} audio pairs into RAM...")
                self.cached_mix = []
                self.cached_sources = [[] for _ in range(len(self.sources))]
                for i in range(self.length):
                    # Load full mixture
                    x, _ = sf.read(self.mix[i][0], dtype="float32")
                    self.cached_mix.append(x)
                    # Load sources
                    for s_idx, src in enumerate(self.sources):
                        s, _ = sf.read(src[i][0], dtype="float32")
                        self.cached_sources[s_idx].append(s)
                print_(f"[EchoSet] Successfully preloaded {self.length} pairs into RAM.")

    def __len__(self):
        return self.length

    def preprocess_audio_only(self, idx: int):
        if self.n_src == 1:
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len
            # Load mixture
            x, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            # Load sources
            s, _ = sf.read(
                self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            # torch from numpy
            target = torch.from_numpy(s)
            mixture = torch.from_numpy(x)
            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                target = normalize_tensor_wav(target, eps=self.EPS, std=m_std)
            # return mixture, target.unsqueeze(0), self.mix[idx][0].split("/")[-1]
            return mixture, target.unsqueeze(0), self.mix[idx][0]
        # import pdb; pdb.set_trace()
        if self.n_src == 2:
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len
            # Load mixture and sources (from RAM cache if enabled, else disk)
            if self.in_memory and self.cached_mix is not None:
                x = self.cached_mix[idx][rand_start:stop] if stop is not None else self.cached_mix[idx][rand_start:]
                source_arrays = [
                    self.cached_sources[s_idx][idx][rand_start:stop] if stop is not None else self.cached_sources[s_idx][idx][rand_start:]
                    for s_idx in range(len(self.sources))
                ]
            else:
                x, _ = sf.read(
                    self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
                )
                source_arrays = []
                for src in self.sources:
                    s, _ = sf.read(
                        src[idx][0], start=rand_start, stop=stop, dtype="float32"
                    )
                    source_arrays.append(s)
            sources = torch.from_numpy(np.vstack(source_arrays))
            mixture = torch.from_numpy(x)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            # return mixture, sources, self.mix[idx][0].split("/")[-1]
            return mixture, sources, self.mix[idx][0]

    def __getitem__(self, index: int):
        return self.preprocess_audio_only(index)


class EchoSetDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        subset_percentile: float = None,
        subset_mode: str = "close",
        subset_fixed_samples: int = None,
        subset_seed: int = 42,
        subset_range: tuple = None,
        in_memory: bool = False,
    ) -> None:
        super().__init__()
        self.in_memory = in_memory
        self.cached_mix = None
        self.cached_sources = None
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        # this line allows to access init params with 'self.hparams' attribute
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.subset_percentile = subset_percentile
        self.subset_mode = subset_mode
        self.subset_fixed_samples = subset_fixed_samples
        self.subset_seed = subset_seed
        self.subset_range = subset_range
        self.subset_range = subset_range
        
        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = MP3DDataset(
            json_dir=self.train_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            subset_percentile=self.subset_percentile,
            subset_mode=self.subset_mode,
            subset_fixed_samples=self.subset_fixed_samples,
            subset_seed=self.subset_seed,
            subset_range=self.subset_range,
            in_memory=self.in_memory,
        )
        self.data_val = MP3DDataset(
            json_dir=self.valid_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=None,
            normalize_audio=self.normalize_audio,
        )
        self.data_test = MP3DDataset(
            json_dir=self.test_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=None,
            normalize_audio=self.normalize_audio,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test
