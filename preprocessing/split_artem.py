"""Import Artem's full structure-ID assignments into converted DeePTB sets.

Run from the repository root; see docs/artem_path_benchmark.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversion.sampling import dataset_sampling
from preprocessing.split_train_val import (
    discover_sets, materialize_set, path_is_within, prepare_output_dirs,
    write_manifests,
)

KEYS = ('train_selected', 'val_selected', 'train_removed', 'val_removed')


def load_assignments(path, profile):
    raw = json.loads(path.read_text(encoding='utf-8'))
    seen = set()
    for key in KEYS:
        values = raw.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f'{key}: expected a list of nonempty structure IDs')
        for value in values:
            if value in seen:
                raise ValueError(f'Duplicate structure ID across manifest lists: {value}')
            seen.add(value)
    assignments = {}
    for key in KEYS:
        if profile == 'selected' and key.endswith('removed'):
            continue
        for value in raw[key]:
            assignments[value] = 'train' if key.startswith('train') else 'val'
    return assignments, seen


def map_sets(root, assignments, known, sampling):
    """Match exact path components, never hashes, substrings, or local ordinals."""
    mapped, omitted, wrong_sampling = {}, [], []
    for folder in discover_sets(root):
        report_path = folder / 'conversion_report.json'
        if not report_path.is_file():
            raise ValueError(f'{folder}: conversion_report.json required for source identity')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        source = report.get('source_case')
        if not isinstance(source, str) or not source:
            raise ValueError(f'{folder}: missing source_case')
        matches = set(source.replace('\\', '/').split('/')) & known
        if len(matches) > 1:
            raise ValueError(f'{folder}: ambiguous source identity: {sorted(matches)}')
        identity = next(iter(matches), None)
        if identity not in assignments:
            omitted.append({'set': folder.name, 'source': source,
                            'reason': 'excluded_by_profile' if identity else 'unmapped_source'})
            continue
        if dataset_sampling(folder) != sampling:
            wrong_sampling.append({'set': folder.name, 'structure_id': identity})
            continue
        if identity in mapped:
            raise ValueError(f'Duplicate converted structure: {identity}')
        mapped[identity] = {'set': folder.name, 'structure_id': identity,
                            'source_case': source, 'split': assignments[identity]}
    return mapped, omitted, wrong_sampling


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_root', type=Path)
    p.add_argument('output_root', type=Path)
    p.add_argument('--manifest', type=Path, required=True,
                   help='Local Artem data_split.json; structure IDs are not bundled with this script')
    p.add_argument('--profile', choices=('selected', 'all'), default='selected',
                   help='selected: original benchmark; all: restore removed IDs to their recorded sides')
    p.add_argument('--sampling', choices=('path', 'mesh'), default='path')
    p.add_argument('--mode', choices=('copy', 'symlink'), default='copy')
    p.add_argument('--allow-missing', action='store_true',
                   help='Explicitly permit an available-data subset; not an exact manifest reproduction')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    root, output = args.input_root.resolve(), args.output_root.resolve()
    if path_is_within(output, root) or path_is_within(root, output):
        p.error('Input and output roots must be separate, non-nested directories')
    assignments, known = load_assignments(args.manifest, args.profile)
    mapped, omitted, wrong_sampling = map_sets(root, assignments, known, args.sampling)
    missing = sorted(set(assignments) - mapped.keys())
    records = sorted(mapped.values(), key=lambda r: int(r['set'][4:]))
    train = [r['set'] for r in records if r['split'] == 'train']
    val = [r['set'] for r in records if r['split'] == 'val']
    audit = {'profile': args.profile, 'sampling': args.sampling,
             'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
             'exact_manifest_membership': not missing,
             'n_train': len(train), 'n_val': len(val), 'n_missing': len(missing),
             'missing_structure_ids': missing, 'omitted_sets': omitted,
             'wrong_sampling_sets': wrong_sampling, 'records': records}
    print(json.dumps(audit, indent=2))
    if args.dry_run:
        return
    if missing and not args.allow_missing:
        p.error(f'{len(missing)} assigned structures missing; inspect --dry-run. '
                'Use --allow-missing only for a documented subset experiment.')
    if not train or not val:
        p.error('At least one mapped structure is required on each side')
    if output.exists():
        p.error('Output already exists; choose a fresh root to preserve previous results')
    train_dir, val_dir = prepare_output_dirs(output, overwrite=False)
    for record in records:
        dest = train_dir if record['split'] == 'train' else val_dir
        materialize_set(root / record['set'], dest / record['set'], args.mode)
    selection = {'method': 'artem_manifest', 'profile': args.profile,
                 'kpoint_sampling': args.sampling,
                 'manifest_sha256': audit['manifest_sha256'],
                 'exact_manifest_membership': not missing}
    write_manifests(output, input_root=root, mode=args.mode, selection=selection,
                    train_names=train, val_names=val)
    (output / 'artem_split_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (output / 'artem_data_split.json').write_bytes(args.manifest.read_bytes())
    print(f'Wrote {len(train)} training / {len(val)} validation sets to {output}')


if __name__ == '__main__':
    main()
