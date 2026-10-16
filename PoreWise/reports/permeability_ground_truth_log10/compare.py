"""Compare saved directional permeability against the uploaded ground truth."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RESULTS = ROOT / 'porewise/results'
SOURCE = RESULTS / 'ground_truth.xlsx'
truth = pd.read_excel(SOURCE, sheet_name='Sheet1')
truth['sample'] = truth['filename'].map(lambda name: Path(str(name)).stem)
indexed = truth.set_index('sample')
rows = []
matched = set()
for file in sorted(RESULTS.glob('*/result.json')):
    result = json.loads(file.read_text())
    sample = file.parent.name
    reference = indexed.loc[sample]
    matched.add(sample)
    for i, axis in enumerate('xyz'):
        actual = float(reference[f'permeability_{axis}_m2'])
        saved = result['K_m2'][i][i]
        prediction = float(saved) if saved is not None else np.nan
        load = result['loads'][i]
        valid = bool(result['valid_columns'][i] and load['valid_for_permeability'])
        reason = ''
        if not valid:
            reason = f"Invalid directional solve: {load['termination_reason']}"
        elif not np.isfinite(actual) or actual <= 0:
            reason = 'Ground truth is not finite and positive'
        elif not np.isfinite(prediction) or prediction <= 0:
            reason = 'Computed permeability is not finite and positive'
        included = not reason
        log_actual = np.log10(actual) if np.isfinite(actual) and actual > 0 else np.nan
        log_prediction = np.log10(prediction) if np.isfinite(prediction) and prediction > 0 else np.nan
        rows.append(dict(sample=sample, filename=reference['filename'], axis=axis,
                         ground_truth_m2=actual, computed_m2=prediction,
                         log10_ground_truth=log_actual, log10_computed=log_prediction,
                         log10_error=log_prediction-log_actual,
                         computed_to_truth_ratio=prediction/actual if actual > 0 else np.nan,
                         directional_valid=valid, tensor_valid=result['valid_for_permeability'],
                         included_directional=included, exclusion_reason=reason,
                         termination_reason=load['termination_reason'],
                         iterations=load['iterations_completed'], result_file=str(file)))
comparisons = pd.DataFrame(rows)
complete_samples = set(comparisons.groupby('sample').filter(
    lambda group: len(group) == 3 and group['included_directional'].all()
    and group['tensor_valid'].all())['sample'])
comparisons['included_complete_tensor'] = comparisons['sample'].isin(complete_samples)
metrics = []
for cohort, flag in [('complete_tensors', 'included_complete_tensor'),
                     ('valid_directions', 'included_directional')]:
    for axis in 'xyz':
        subset = comparisons[(comparisons.axis == axis) & comparisons[flag]]
        observed = subset.log10_ground_truth.to_numpy()
        predicted = subset.log10_computed.to_numpy()
        residual = predicted - observed
        sse = np.sum(residual**2)
        sst = np.sum((observed-observed.mean())**2)
        metrics.append(dict(cohort=cohort, axis=axis.upper(), n=len(subset),
                            r2=1-sse/sst if sst > 0 else np.nan,
                            mae_log10=float(np.mean(np.abs(residual))),
                            rmse_log10=float(np.sqrt(np.mean(residual**2)))))
metrics = pd.DataFrame(metrics)
uncomputed = truth[~truth['sample'].isin(matched)][
    ['filename', 'sample', 'permeability_x_m2', 'permeability_y_m2', 'permeability_z_m2']]
excluded = comparisons[~comparisons.included_directional]
method = {
    'ground_truth_file': str(SOURCE),
    'ground_truth_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'sheet': 'Sheet1', 'ground_truth_rows': len(truth), 'matched_samples': len(matched),
    'uncomputed_reference_samples': len(uncomputed), 'complete_tensor_samples': len(complete_samples),
    'log_transform': 'log10 of the numerical permeability in m^2, equivalently log10(k / 1 m^2)',
    'direction_mapping': 'X=K_m2[0][0], Y=K_m2[1][1], Z=K_m2[2][2]; components xyz',
    'matching': 'Exact filename stem, removing the .tif extension',
    'r2': '1 - sum((prediction_log10 - truth_log10)^2) / sum((truth_log10 - mean(truth_log10))^2); no regression fitted',
    'mae': 'mean(abs(prediction_log10 - truth_log10))',
    'rmse': 'sqrt(mean((prediction_log10 - truth_log10)^2))',
    'error_units': 'dex (base-10 log units), not m^2; R2 is dimensionless',
    'complete_tensors': 'Common samples with accepted full tensors and three positive finite reference/prediction pairs',
    'valid_directions': 'Each direction separately; accepted finite positive directional results, including valid columns from incomplete tensors',
    'exclusions': 'Invalid solves and nonpositive/nonfinite pairs are excluded; no arbitrary outlier removal or missing-value imputation',
}
for name, frame in [('metrics',metrics),('comparisons',comparisons),('excluded_directions',excluded),('uncomputed_ground_truth',uncomputed)]:
    frame.to_csv(OUT/f'{name}.csv', index=False)
(OUT/'methodology.json').write_text(json.dumps(method, indent=2)+'\n')
with pd.ExcelWriter(OUT/'permeability_comparison.xlsx', engine='openpyxl') as writer:
    metrics.to_excel(writer, sheet_name='log10_metrics', index=False)
    comparisons.to_excel(writer, sheet_name='matched_comparisons', index=False)
    excluded.to_excel(writer, sheet_name='excluded_directions', index=False)
    uncomputed.to_excel(writer, sheet_name='uncomputed_ground_truth', index=False)
    pd.DataFrame([{'item':key, 'value':value} for key,value in method.items()]).to_excel(
        writer, sheet_name='methodology', index=False)
    for sheet in writer.book.worksheets:
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = min(
                65, max(14, max(len(str(cell.value or '')) for cell in column)+2))

fig, axes = plt.subplots(1,3,figsize=(13,4.3), layout='constrained')
for axis, panel in zip('xyz', axes):
    complete = comparisons[(comparisons.axis == axis) & comparisons.included_complete_tensor]
    extra = comparisons[(comparisons.axis == axis) & comparisons.included_directional
                        & ~comparisons.included_complete_tensor]
    panel.scatter(complete.log10_ground_truth,complete.log10_computed,s=35,label='21 complete tensors')
    if len(extra):
        panel.scatter(extra.log10_ground_truth,extra.log10_computed,s=70,marker='x',
                      color='tab:red',label='69.952: valid direction')
    valid = comparisons[(comparisons.axis == axis) & comparisons.included_directional]
    bounds = [min(valid.log10_ground_truth.min(),valid.log10_computed.min())-.15,
              max(valid.log10_ground_truth.max(),valid.log10_computed.max())+.15]
    panel.plot(bounds,bounds,'--',color='gray',linewidth=1,label='Perfect agreement')
    panel.set(xlim=bounds,ylim=bounds,xlabel='Ground truth: log10(k / 1 m²)',
              ylabel='PoreWise: log10(k / 1 m²)',title=f'{axis.upper()} direction')
    panel.set_aspect('equal',adjustable='box')
    panel.grid(alpha=.2)
    panel.legend(fontsize=8)
fig.savefig(OUT/'log10_comparison.png',dpi=180)
plt.close(fig)
print(metrics.to_string(index=False))
print('\nCoverage:',json.dumps({k:method[k] for k in ['ground_truth_rows','matched_samples','uncomputed_reference_samples','complete_tensor_samples']}))
print('\nAdditional valid pairs:')
print(comparisons[comparisons.included_directional & ~comparisons.included_complete_tensor][
    ['sample','axis','ground_truth_m2','computed_m2','log10_error']].to_string(index=False))
print('\nSaved:',OUT/'permeability_comparison.xlsx')
