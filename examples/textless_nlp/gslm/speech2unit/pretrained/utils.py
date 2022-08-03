# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import random
import shutil
import io
import yt
import yt.wrapper
import numpy as np
from transformers import Wav2Vec2Processor, HubertModel, Wav2Vec2FeatureExtractor
import soundfile as sf

import torch
import torch.nn.functional as F
import tqdm
from examples.textless_nlp.gslm.speech2unit.pretrained.cpc_feature_reader import (
    CpcFeatureReader,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.hubert_feature_reader import (
    HubertFeatureReader,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.logmel_feature_reader import (
    LogMelFeatureReader,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.w2v2_feature_reader import (
    Wav2VecFeatureReader,
)


def get_feature_reader(feature_type):
    if feature_type == "logmel":
        return LogMelFeatureReader
    elif feature_type == "hubert":
        return HubertFeatureReader
    elif feature_type == "w2v2":
        return Wav2VecFeatureReader
    elif feature_type == "cpc":
        return CpcFeatureReader
    else:
        raise NotImplementedError(f"{feature_type} is not supported.")


def get_feature_iterator(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, yt_token, norm, mode='classic'
):
    if mode == 'classic':
        feature_reader_cls = get_feature_reader(feature_type)
        with open(manifest_path, "r") as fp:
            lines = fp.read().split("\n")
            root = lines.pop(0).strip()
            file_path_list = [
                os.path.join(root, line.split("\t")[0])
                for line in lines
                if len(line) > 0
            ]
            if sample_pct < 1.0:
                file_path_list = random.sample(
                    file_path_list, int(sample_pct * len(file_path_list))
                )
            num_files = len(file_path_list)
            reader = feature_reader_cls(
                checkpoint_path=checkpoint_path, layer=layer
            )

            def iterate():
                for file_path in file_path_list:
                    feats = reader.get_feats(file_path)
                    yield feats.cpu().numpy()
    else:
        yt_client = yt.wrapper.YtClient('hahn', token=yt_token)
        num_files = yt_client.row_count(manifest_path)

        if mode == 'huggingface-yt':

            if norm == 'true':
                processor = Wav2Vec2FeatureExtractor()#Wav2Vec2Processor.from_pretrained(checkpoint_path)
            
            model = HubertModel.from_pretrained(checkpoint_path)
            if torch.cuda.is_available():
                model = model.cuda()
            model = model.eval()

            def iterate():
                table_sh = yt_client.read_table(manifest_path)
                for row in table_sh:
                    with torch.no_grad():
                        data, sr = sf.read(io.BytesIO(yt.yson.get_bytes(row['pcm__wav'])))
                        assert sr  == 16000, "File must have sample rate equal 16k"
                        if norm == 'true':
                            input_values = processor(data, return_tensors="pt", sampling_rate=16000).input_values
                        else:
                            input_values = torch.tensor(data)
                            input_values = input_values.view(1, -1).float() 
                        if torch.cuda.is_available():
                            input_values = input_values.to("cuda")
                        feats = torch.squeeze(model(input_values, output_hidden_states=True).hidden_states[layer].detach(), 0)
                    yield row['ID'], feats.cpu().numpy()
        elif mode == 'chpt-yt':
            feature_reader_cls = get_feature_reader(feature_type)
            reader = feature_reader_cls(
                checkpoint_path=checkpoint_path, layer=layer
            )
            def iterate():
                table_sh = yt_client.read_table(manifest_path)
                for row in table_sh:
                    feats = reader.get_feats(io.BytesIO(yt.yson.get_bytes(row['pcm__wav'])))
                    yield row['ID'], feats.cpu().numpy()
        else:
            raise ValueError("Некорректное значение параметра mode")


    return iterate, num_files


def get_features(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, flatten, yt_token, norm, mode
):
    generator, num_files = get_feature_iterator(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        yt_token=yt_token,
        norm=norm,
        mode=mode
    )
    iterator = generator()


    features_list = []
    ids_list = []
    for features in tqdm.tqdm(iterator, total=num_files):
        if mode == 'classic':
            features_list.append(features)
        else:
            features_list.append(features[1])
            ids_list.append(features[0])

    # Explicit clean up
    del iterator
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    if flatten:
        return np.concatenate(features_list)
    
    if mode == 'classic':
        return features_list
    return ids_list, features_list


def get_and_dump_features(
    feature_type,
    checkpoint_path,
    layer,
    manifest_path,
    sample_pct,
    flatten,
    out_features_path,
):
    # Feature extraction
    features_batch = get_features(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        flatten=flatten,
    )

    # Save features
    out_dir_path = os.path.dirname(out_features_path)
    os.makedirs(out_dir_path, exist_ok=True)
    shutil.copyfile(
        manifest_path,
        os.path.join(out_dir_path, os.path.basename(manifest_path)),
    )
    np.save(out_features_path, features_batch)

    return features_batch
