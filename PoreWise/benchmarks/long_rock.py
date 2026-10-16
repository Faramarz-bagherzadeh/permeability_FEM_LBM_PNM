"""Execute the reviewed long-rock configuration after owner run-budget approval."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from porewise.solver import periodic
from porewise.tensor import assemble_tensor
from porewise.io import load_image,save_result,write_json,provenance
from porewise.backends import cp


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='benchmarks/configs/long_rock.json');p.add_argument('--output',default='results/bentheimer-long')
 p.add_argument('--reuse-x',help='standard-tolerance x load from convergence_review')
 a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if a.reuse_x:cfg['reuse_x_result']=a.reuse_x
 m=load_image(cfg['mask'])
 write_json(out/'configuration.json',cfg)
 used=json.loads((out/'budget.json').read_text())['charged_seconds'] if (out/'budget.json').exists() else 0.
 budget=cfg['additional_simulation_budget_seconds']
 for fi,F in enumerate(cfg['forces']):
  loads=[];force_time=0.
  for j in range(3):
   directory=out/f'force_{fi}_load_{j}'
   if (directory/'result.json').exists():
    prior=json.loads((directory/'result.json').read_text())
    loads.append(prior);force_time+=prior['elapsed_s'];continue
   if fi==0 and j==0 and a.reuse_x:
    prior=json.loads(Path(a.reuse_x).read_text())
    # Different maximum-step/time caps do not change an already accepted state.
    # Keep its actual settings and elapsed cost; do not charge the same run twice.
    save_result(directory,prior,dict(configuration=cfg,reused_from=a.reuse_x,
     original_metadata=prior['metadata'],cost_note='included in tensor elapsed; charged previously to convergence review'))
    loads.append(prior);force_time+=prior['elapsed_s'];continue
   remaining=budget-used
   if remaining<=0:
    write_json(directory/'deferred.json',dict(reason='additional approved budget exhausted'));break
   cp.get_default_memory_pool().free_all_blocks();free,total=cp.cuda.runtime.memGetInfo()
   force=np.zeros(3);force[j]=F;s=dict(cfg['settings']);s['wall_timeout_s']=min(s['wall_timeout_s'],remaining)
   t=time.perf_counter()
   try:r=periodic(m,force,**s)
   except Exception as e:
    write_json(directory/'failure.json',dict(reason=repr(e),configuration=cfg));raise
   cost=time.perf_counter()-t;used+=cost;force_time+=cost;loads.append(r)
   save_result(directory,r,dict(configuration=cfg,reference_status='application only; no independent reference comparison',resources_before=dict(gpu_free_bytes=free)))
   write_json(out/'budget.json',dict(charged_seconds=used,limit_seconds=budget))
  if len(loads)==3:
   tensor=assemble_tensor(loads,cfg['voxel_size_m']);tensor['elapsed_s']=force_time
   save_result(out/f'tensor_force_{fi}',tensor,dict(configuration=cfg))
 if len(cfg['forces'])==2:
  p0=out/'tensor_force_0/result.json';p1=out/'tensor_force_1/result.json'
  if p0.exists() and p1.exists():
   r0=json.loads(p0.read_text());r1=json.loads(p1.read_text());valid=r0['valid_for_permeability'] and r1['valid_for_permeability']
   err=float(np.linalg.norm(np.array(r0['K_lu'])-r1['K_lu'])/np.linalg.norm(r0['K_lu'])) if valid else None
   write_json(out/'force_sensitivity.json',dict(valid_pair=valid,relative_tensor_error=err,acceptance_threshold=.01,passed=valid and err<.01))

if __name__=='__main__':main()
