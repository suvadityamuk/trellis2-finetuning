"""Generic adapter for the TRELLIS.2 `data_toolkit`: any folder of 3D meshes.

Every file under `--mesh_root` whose extension Blender can import (see
`blender_script/dump_mesh.py`) becomes one object. Layout is free-form: flat
(`chair.glb`, `table.glb`) or nested (`<id>/meshes/model.obj` with its `.mtl` and
textures beside it). The object id is the path relative to `--mesh_root` without
the extension, with `/` replaced by `__`.

Like the GSO adapter, this only indexes files that are already on disk, so
`download` is a no-op.
"""
import os
import argparse
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm
from utils import get_file_hash

MESH_EXTENSIONS = ('glb', 'gltf', 'obj', 'fbx', 'ply', 'stl', 'usd', 'usda', 'usdz', 'blend')


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--mesh_root', type=str, required=True,
                        help='Directory searched recursively for mesh files')
    parser.add_argument('--mesh_extensions', type=str, default=','.join(MESH_EXTENSIONS),
                        help='Comma-separated extensions to index')
    parser.add_argument('--max_objects', type=int, default=None,
                        help='Index only the first N meshes (sorted by path)')
    parser.add_argument('--aesthetic_score', type=float, default=6.0,
                        help='Constant score for every object; the trainers filter on >= 4.5')


def find_meshes(root: str, extensions: str) -> list[str]:
    exts = {e.strip().lower().lstrip('.') for e in extensions.split(',') if e.strip()}
    paths = (p for p in glob(os.path.join(root, '**', '*'), recursive=True) if os.path.isfile(p))
    return sorted(p for p in paths if p.rsplit('.', 1)[-1].lower() in exts)


def get_metadata(**kwargs) -> pd.DataFrame:
    root = os.path.abspath(kwargs['mesh_root'])
    paths = find_meshes(root, kwargs.get('mesh_extensions') or ','.join(MESH_EXTENSIONS))
    if kwargs.get('max_objects'):
        paths = paths[:kwargs['max_objects']]
    score = kwargs.get('aesthetic_score', 6.0)

    def record(path: str) -> dict:
        rel = os.path.splitext(os.path.relpath(path, root))[0]
        return {'sha256': get_file_hash(path), 'file_identifier': rel.replace(os.sep, '__'),
                'local_path': path, 'aesthetic_score': score}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        records = list(tqdm(executor.map(record, paths), total=len(paths), desc='Indexing meshes'))
    return pd.DataFrame.from_records(records).drop_duplicates('sha256')


def download(metadata, **kwargs) -> pd.DataFrame:
    columns = [c for c in ('sha256', 'local_path') if c in metadata.columns]
    return metadata[columns] if columns else pd.DataFrame(columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, no_file=False, desc='Processing objects') -> pd.DataFrame:
    metadata = metadata.to_dict('records')
    records = []
    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
         tqdm(total=len(metadata), desc=desc) as pbar:
        def worker(metadatum):
            try:
                file = None if no_file else os.path.join(output_dir, metadatum['local_path'])
                record = func(file, metadatum)
                if record is not None:
                    records.append(record)
            except Exception as e:
                print(f"Error processing object {metadatum.get('sha256')}: {e}")
            pbar.update()
        executor.map(worker, metadata)
        executor.shutdown(wait=True)
    return pd.DataFrame.from_records(records)
