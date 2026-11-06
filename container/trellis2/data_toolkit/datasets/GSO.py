"""Google Scanned Objects (GSO) adapter for the TRELLIS.2 `data_toolkit`.

The Hub dataset (`suvadityamuk/google-scanned-objects`) is a WebDataset: every
object is a group of tar members named `<id>.obj`, `<id>.mtl`, `<id>.texture.png`,
`<id>.glb`, `<id>.json`, ... . `jobs/run_pilot.sh` materializes the three files we
need per object into

    <gso_root>/<object_id>/meshes/{model.obj, model.mtl, texture.png}

which is exactly what Blender expects: the OBJ says `mtllib model.mtl` and the MTL
says `map_Kd texture.png`, so the diffuse map binds when both sit beside the OBJ.

This adapter only *indexes* that on-disk layout — the meshes are already local, so
`download` is a no-op and `get_metadata` just points `local_path` at each OBJ.
"""
import os
import argparse
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm
from utils import get_file_hash

# gso-dataset/_src lives at the repo root, three levels above this file.
DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'gso-dataset', '_src'))
AESTHETIC_SCORE = 6.0  # GSO ships no scores; a constant clears the >= 4.5 filter.


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--gso_root', type=str, default=DEFAULT_ROOT,
                        help='Directory of materialized GSO objects (<id>/meshes/model.obj)')


def _record(obj_path: str) -> dict:
    """One metadata row for a materialized object; sha256 keys every later stage."""
    object_id = os.path.basename(os.path.dirname(os.path.dirname(obj_path)))
    return {
        'sha256': get_file_hash(obj_path),
        'file_identifier': object_id,
        'local_path': os.path.abspath(obj_path),
        'aesthetic_score': AESTHETIC_SCORE,
    }


def get_metadata(**kwargs) -> pd.DataFrame:
    root = kwargs.get('gso_root') or DEFAULT_ROOT
    objs = sorted(glob(os.path.join(root, '*', 'meshes', 'model.obj')))

    records = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
         tqdm(total=len(objs), desc='Indexing GSO') as pbar:
        def worker(obj_path):
            try:
                records.append(_record(obj_path))
            except Exception as e:
                print(f'Error indexing {obj_path}: {e}')
            pbar.update()
        executor.map(worker, objs)
        executor.shutdown(wait=True)
    return pd.DataFrame.from_records(records)


def download(metadata, **kwargs) -> pd.DataFrame:
    # Meshes are already local (materialized by run_pilot.sh); nothing to fetch.
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
