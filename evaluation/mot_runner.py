"""Explicit-manifest manual-frame MOT evaluation using external TrackEval."""

import configparser
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def fingerprint(paths):
    result = {}
    for path in sorted(set(paths)):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                digest.update(block)
        result[str(path)] = digest.hexdigest()
    return result


def calculate(work, method, trackeval):
    evaluator = trackeval.Evaluator(dict(USE_PARALLEL=False, NUM_PARALLEL_CORES=1,
        BREAK_ON_ERROR=True, RETURN_ON_ERROR=False, LOG_ON_ERROR=None,
        PRINT_RESULTS=False, PRINT_ONLY_COMBINED=True, PRINT_CONFIG=False,
        TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True, OUTPUT_SUMMARY=True,
        OUTPUT_EMPTY_CLASSES=True, OUTPUT_DETAILED=True, PLOT_CURVES=False))
    dataset = trackeval.datasets.MotChallenge2DBox(dict(
        GT_FOLDER=str(work/'gt'), TRACKERS_FOLDER=str(work/'trackers'),
        OUTPUT_FOLDER=str(work/'output'), TRACKERS_TO_EVAL=[method],
        CLASSES_TO_EVAL=['pedestrian'], BENCHMARK='SpaceAnimal', SPLIT_TO_EVAL='sparse_gt',
        INPUT_AS_ZIP=False, PRINT_CONFIG=False, DO_PREPROC=False,
        TRACKER_SUB_FOLDER='data', OUTPUT_SUB_FOLDER='', SEQMAP_FILE=str(work/'seqmap.txt'),
        SKIP_SPLIT_FOL=True))
    metric_config = dict(THRESHOLD=.5, PRINT_CONFIG=False)
    results, messages = evaluator.evaluate([dataset], [trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(metric_config), trackeval.metrics.Identity(metric_config)])
    if messages[dataset.get_name()][method] != 'Success':
        raise RuntimeError(f'TrackEval failed for {method}')
    combined = results[dataset.get_name()][method]['COMBINED_SEQ']['pedestrian']
    return dict(HOTA=float(np.mean(combined['HOTA']['HOTA'])*100),
                DetA=float(np.mean(combined['HOTA']['DetA'])*100),
                AssA=float(np.mean(combined['HOTA']['AssA'])*100),
                MOTA=float(combined['CLEAR']['MOTA']*100),
                IDF1=float(combined['Identity']['IDF1']*100),
                IDs=int(combined['CLEAR']['IDSW']))


def run_mot(args, *, safe_name, unused_output, methods_from_file, filtered_mot,
            load_manual_frame_ids):
    """Keep data selection explicit; never infer an exclusion from a score."""
    output = unused_output(args.output_dir)
    dataset, results_root = args.dataset_root.resolve(), args.results_root.resolve()
    sources = [args.manifest.resolve(), args.methods_file.resolve(), Path(__file__).resolve(),
               Path(__file__).resolve().parents[1]/'evaluation/cli.py']
    before = fingerprint(sources)

    def remember(*paths):
        sources.extend(paths)
        before.update(fingerprint(paths))

    methods, roots = methods_from_file(args.methods_file, results_root)
    if any(output == root or root in output.parents for root in (dataset, results_root, *roots.values())):
        raise ValueError('MOT output must be outside all input data/result roots')
    species = safe_name(args.species, 'species')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    sequences = manifest['species'][species]['val']
    if not isinstance(sequences, list) or not sequences or len(set(sequences)) != len(sequences):
        raise ValueError('explicit test sequence list must be non-empty and unique')
    prepared = {}
    for name in sequences:
        safe_name(name, 'sequence')
        root = dataset/species/'val'/name
        info = root/'seqinfo.ini'
        remember(info)
        config = configparser.ConfigParser()
        if not config.read(info):
            raise FileNotFoundError(info)
        length = config.getint('Sequence', 'seqLength')
        if length < 1:
            raise ValueError(f'{name}: invalid sequence length')
        gt = root/'gt/gt_manual.txt'
        if not gt.is_file():
            gt = root/'gt/gt.txt'
        remember(gt)
        mask = root/'gt/manual_frame_ids.txt'
        if mask.is_file():
            remember(mask)
            manual = load_manual_frame_ids(mask)
            mask_source = str(mask)
        else:
            manual = {int(float(line.split(',')[0])) for line in gt.read_text().splitlines() if line.strip()}
            mask_source = 'frames present in sparse GT (original protocol)'
        if not manual or min(manual) < 1 or max(manual) > length:
            raise ValueError(f'{name}: manual mask lies outside sequence')
        gt_text = filtered_mot(gt, manual, length)
        # A dense reference without a separate manual GT file is ambiguous;
        # refuse rather than silently using interpolated boxes for MOT.
        source_rows = [line for line in gt.read_text().splitlines() if line.strip()]
        if len(gt_text.splitlines()) != len(source_rows):
            raise ValueError(f'{name}: provide sparse manual GT, not interpolated reference')
        if any(len(line.split(',')) < 9 for line in source_rows):
            raise ValueError(f'{name}: MOT ground truth requires nine fields')
        predictions = {}
        for method in methods:
            source = roots[method.directory]/species/method.directory/f'{name}.txt'
            remember(source)
            predictions[method.directory] = filtered_mot(source, manual, length)
        prepared[name] = dict(gt=gt_text, info=info, predictions=predictions,
                              manual_frames=sorted(manual), mask_source=mask_source)
    trackeval_root = args.trackeval_root.resolve()
    if not (trackeval_root/'trackeval/__init__.py').is_file():
        raise FileNotFoundError('supply a TrackEval source checkout with --trackeval-root')
    upstream_sources = sorted((trackeval_root/'trackeval').rglob('*.py'))
    upstream_before = fingerprint(upstream_sources)
    if before != fingerprint(sources):
        raise RuntimeError('evaluation input changed during preparation')
    sys.path.insert(0, str(trackeval_root))
    for name, replacement in dict(float=float, int=int, bool=bool).items():
        if name not in np.__dict__:
            setattr(np, name, replacement)
    with contextlib.redirect_stdout(io.StringIO()):
        import trackeval
    if not Path(trackeval.__file__).resolve().is_relative_to(trackeval_root):
        raise RuntimeError('a different TrackEval has already been imported')
    values = {}
    for method in methods:
        work = output/method.directory
        for name, item in prepared.items():
            target = work/'gt'/name/'gt'
            target.mkdir(parents=True)
            target.joinpath('gt.txt').write_text(item['gt'], encoding='utf-8')
            shutil.copyfile(item['info'], target.parent/'seqinfo.ini')
            pred = work/'trackers'/method.directory/'data'
            pred.mkdir(parents=True, exist_ok=True)
            pred.joinpath(f'{name}.txt').write_text(item['predictions'][method.directory], encoding='utf-8')
        work.joinpath('seqmap.txt').write_text('name\n' + ''.join(f'{name}\n' for name in sequences))
        values[method.label] = calculate(work, method.directory, trackeval)
    if before != fingerprint(sources) or upstream_before != fingerprint(upstream_sources):
        raise RuntimeError('evaluation inputs or executed TrackEval changed during evaluation')
    payload = dict(species=species, sequences=sequences, methods=values,
        metrics_unit='percent except IDs', IDs='CLEAR IDSW',
        clear_identity_iou_threshold=.5, hota_thresholds='official 0.05:0.05:0.95 average')
    provenance = dict(input_files=before, inputs_unchanged=True,
        trackeval_root=str(trackeval_root), trackeval_sources_sha256=upstream_before,
        frame_protocol='original manual frames only, without interpolation or renumbering',
        sequences={name: {k: item[k] for k in ('manual_frames', 'mask_source')}
                   for name, item in prepared.items()})
    output.joinpath('metrics.json').write_text(json.dumps(payload, indent=2)+'\n')
    output.joinpath('provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'calculated manual-frame MOT metrics for {species}: {output / "metrics.json"}')
