"""Recover rounded, complete per-version dev checks from preserved trace pages.
Run when original experiment JSONL logs are unavailable. Never reconstruct missing metrics.
"""
from pathlib import Path
import re,html,json
r=Path(__file__).resolve().parent
root=r.parent.parent
out={}
for p in sorted((root/'docs/mitigating-reward-hacking/runs').glob('*.html')):
 s=p.read_text();summary=re.search(r'<section class="ch" id="final">[\s\S]*?</section>',s).group()
 tips={int(re.match(r'Dev check (\d+)',html.unescape(t))[1]):html.unescape(t) for t in re.findall('data-tip="([^"]+)"',s) if t.startswith('Dev check')}
 checks=[]
 for row in re.findall(r'<tr><td><a href="#check-[\s\S]*?</tr>',summary):
  cells=[html.unescape(re.sub('<[^>]+>','',x)).strip() for x in re.findall(r'<td[^>]*>([\s\S]*?)</td>',row)]
  k=int(re.search(r'\d+',cells[0])[0]);tip=tips[k]
  stamp=re.search(r' at (.*?):',tip)[1];hours=sum(float(v)*{'h':1,'m':1/60,'s':1/3600}[unit] for v,unit in re.findall(r'(\d+)([hms])',stamp))
  envs=re.findall(r'(addition|cities_qa|sort) (-?[\d.]+)',tip)
  checks.append({'check':k,'hours':hours,'candidate':cells[1],'score':float(cells[2]),'true_reward':float(cells[3]),'hack_rate':float(cells[4].rstrip('%'))/100,'envs':dict(envs)})
 out[p.stem]={'source':f'runs/{p.name}#final','precision':'Metrics rounded as displayed in preserved trace; one check per code version, largest/latest evaluation per environment. May combine separate launcher runs and seeds.','evaluated_versions':len(checks),'complete_curve':[c for c in checks if set(c['envs'])=={'addition','cities_qa','sort'}]}
(r/'data/trace_dev.json').write_text(json.dumps(out,indent=2)+'\n')
for slug in ['opus-8','opus-4','sol-7','sol-4']:print(slug,'versions',out[slug]['evaluated_versions'],'three-env points',len(out[slug]['complete_curve']))
