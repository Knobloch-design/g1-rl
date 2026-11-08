"""Build one HTML page comparing every evaluated PPO and ES run.

    .venv/bin/python compare_runs.py            # -> logs/comparison/index.html

Reads <run>/eval.csv (from eval_runs.py) for every run under logs/rsl_rl and
logs/es, plus each ES run's progress.csv (its in-training curve). Runs are
grouped into panels by policy size; color = algorithm, dashed = SGD-style
variant, and PPO+Adam on the big network is drawn in every panel as the
reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOTS = [Path("logs/rsl_rl/g1_velocity"), Path("logs/es/g1_velocity")]

# Color follows the algorithm (fixed order from the dataviz reference palette).
ALGO_ORDER = ["ppo", "ars", "openai_es", "pgpe", "cmaes", "sep_cmaes"]
ALGO_NAME = {"ppo": "PPO", "ars": "ARS", "openai_es": "OpenAI-ES", "pgpe": "PGPE",
             "cmaes": "CMA-ES", "sep_cmaes": "sep-CMA-ES"}


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def describe(label: str) -> dict:
    """ppo_mlp512x256x128_sgd -> algo, arch, optimizer."""
    parts = label.split("_")
    if parts[0] == "ppo":
        algo, rest = "ppo", parts[1:]
    elif label.startswith("openai_es"):
        algo, rest = "openai_es", parts[2:]
    elif label.startswith("sep_cmaes"):
        algo, rest = "sep_cmaes", parts[2:]
    else:
        algo, rest = parts[0], parts[1:]
    arch = rest[0]
    opt = rest[1] if len(rest) > 1 else {"ppo": "adam", "openai_es": "adam",
                                         "pgpe": "clipup"}.get(algo, "")
    if arch == "linear":
        group = "linear"
    elif arch == "mlp512x256x128":
        group = "big"
    else:
        group = "small"
    return {"algo": algo, "arch": arch, "optimizer": opt, "group": group}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=Path("logs/comparison/index.html"))
    p.add_argument("--exclude", nargs="*", default=["2026-10-31_18-05-16_tracked"],
                   help="run folders to leave out (default: the 1660 Super run)")
    args = p.parse_args()

    runs = []
    for root in ROOTS:
        for run_dir in sorted(root.glob("*")):
            ev = run_dir / "eval.csv"
            if not ev.exists() or run_dir.name in args.exclude:
                continue
            rows = read_csv(ev)
            if not rows:
                continue
            info = describe(rows[0]["label"])
            nice = ALGO_NAME[info["algo"]]
            if info["optimizer"] and (info["algo"] in ("ppo", "openai_es", "pgpe")):
                nice += f" + {'ClipUp' if info['optimizer'] == 'clipup' else info['optimizer'].upper() if info['optimizer'] == 'sgd' else 'Adam'}"
            nice += f" · {info['arch'].replace('mlp', 'MLP ')}"
            if "lr" in run_dir.name.split("_")[-1]:
                nice += f" ({run_dir.name.split('_')[-1]})"
            progress = []
            if (run_dir / "progress.csv").exists():
                pr = read_csv(run_dir / "progress.csv")
                stride = max(1, len(pr) // 300)
                progress = [{"t": r["wall_time_s"] / 60, "s": r["env_steps"],
                             "y": r["center_return"], "len": r["center_episode_length"]}
                            for r in pr[::stride]]
            num_params = None
            if (run_dir / "config.json").exists():
                num_params = json.loads((run_dir / "config.json").read_text())["num_params"]
            runs.append({"name": run_dir.name, "label": nice, **info,
                         "num_params": num_params, "eval": rows, "progress": progress})

    if not runs:
        raise SystemExit("no eval.csv files found; run eval_runs.py first")
    runs.sort(key=lambda r: (ALGO_ORDER.index(r["algo"]), r["arch"], r["optimizer"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(runs))
    args.out.write_text(html)
    print(f"Wrote {args.out} ({len(runs)} runs)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>G1 Optimizer Comparison</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;
--grid:#e7e6e2;--ref:#8a8984;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#141413;--card:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--grid:#2c2c2a;--ref:#8f8e86;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#141413;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;
--grid:#2c2c2a;--ref:#8f8e86;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 4px}
p.lede,p.note{color:var(--ink2);margin:0 0 16px;max-width:820px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:14px 14px 8px}
.card h3{font-size:14px;margin:0 0 6px;color:var(--ink2);font-weight:600}
.chart{position:relative;height:280px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px}
.tabs button{font:inherit;font-size:13px;padding:4px 12px;border-radius:999px;border:1px solid var(--grid);
background:var(--card);color:var(--ink2);cursor:pointer}
.tabs button[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
.scroll{overflow-x:auto}.sw{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:8px;vertical-align:middle}
</style></head><body><main>
<h1>Training the G1 to walk: PPO vs evolutionary methods</h1>
<p class="lede">Every checkpoint below was replayed by <code>eval_runs.py</code> with the same seeds and no
exploration noise. <b>Return</b> and <b>survival</b> come from 1,024 random-command 20&nbsp;s episodes of the
training task; <b>course falls</b> and <b>tracking error</b> from 64 robots on the fixed 20&nbsp;s course.
Color = algorithm, dashed = SGD variant, gray = PPO + Adam on the big network (the reference).</p>

<div class="tabs" id="xaxis" role="group" aria-label="x axis">
<button aria-pressed="true" data-x="t">vs wall-clock time</button>
<button aria-pressed="false" data-x="s">vs simulated env steps</button></div>

<h2>Episode return</h2><p class="note">Higher is better. Roughly 100 is a policy that walks to command for the full 20 s.</p>
<div class="grid" id="g-return"></div>
<h2>Survival</h2><p class="note">Fraction of the 1,024 robots that never fell in 20 s.</p>
<div class="grid" id="g-surv"></div>
<h2>Course: velocity tracking error</h2><p class="note">m/s between commanded and actual velocity while upright. Lower is better; a robot standing still scores ~0.5.</p>
<div class="grid" id="g-err"></div>
<h2>ES in-training curves</h2><p class="note">Return of each ES run's mean policy during training (every generation, on its own training horizon, so not on the same scale as the evaluation above).</p>
<div class="grid" id="g-train"></div>
<h2>Final results</h2><p class="note">Last checkpoint of each run. "Time to 50" is wall-clock minutes until evaluation return first reached 50.</p>
<div class="card scroll"><table id="tbl"></table></div>
</main>
<script>
const RUNS = __DATA__;
const ORDER = ["ppo","ars","openai_es","pgpe","cmaes","sep_cmaes"];
const GROUPS = [["big","PPO-sized MLP (512-256-128)"],["small","Small MLP (32-32)"],["linear","Linear policy"]];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const color = r => r.name===REF?.name ? css("--ref") : css("--s"+(ORDER.indexOf(r.algo)+1));
const REF = RUNS.find(r=>r.algo==="ppo"&&r.group==="big"&&r.optimizer==="adam");
let X = "t"; const charts = [];
function datasets(group, ykey, fromProgress){
  const rs = RUNS.filter(r=>r.group===group && (!fromProgress || r.progress.length));
  if(!fromProgress && REF && group!=="big") rs.unshift(REF);
  return rs.map(r=>{
    const pts = fromProgress ? r.progress.map(p=>({x: X==="t"?p.t:p.s, y:p.y}))
      : r.eval.map(e=>({x: X==="t"? e.wall_time_s/60 : e.env_steps, y:e[ykey]})).filter(p=>isFinite(p.x));
    const c = color(r);
    return {label:r.label + (r===REF&&group!=="big"?" (reference)":""), data:pts, borderColor:c, backgroundColor:c,
      borderWidth:2, pointRadius: fromProgress?0:3, pointHoverRadius:5, tension:0,
      borderDash: (r.optimizer==="sgd")?[6,4]:[]};
  });
}
function mk(id, ykey, opts={}){
  const host = document.getElementById(id);
  for(const [g,title] of GROUPS){
    if(!RUNS.some(r=>r.group===g && (!opts.progress || r.progress.length))) continue;
    const card = document.createElement("div"); card.className="card";
    card.innerHTML = `<h3>${title}</h3><div class="chart"><canvas></canvas></div>`;
    host.appendChild(card);
    const ch = new Chart(card.querySelector("canvas"), {type:"line",
      data:{datasets:datasets(g, ykey, opts.progress)},
      options:{responsive:true, maintainAspectRatio:false, parsing:false, animation:false,
        interaction:{mode:"nearest", intersect:false},
        scales:{x:{type:"linear", title:{display:true,text:""}, grid:{color:css("--grid")}, ticks:{color:css("--ink2")}},
                y:{title:{display:true,text:opts.ylabel||""}, min:opts.min, max:opts.max,
                   grid:{color:css("--grid")}, ticks:{color:css("--ink2")}}},
        plugins:{legend:{position:"bottom", labels:{color:css("--ink2"), boxWidth:14, boxHeight:2, font:{size:11}}},
          tooltip:{callbacks:{title:items=>X==="t"?`${items[0].parsed.x.toFixed(1)} min`:`${(items[0].parsed.x/1e6).toFixed(0)}M steps`,
            label:i=>`${i.dataset.label}: ${i.parsed.y.toFixed(3)}`}}}}});
    ch._g=g; ch._y=ykey; ch._p=!!opts.progress; charts.push(ch);
  }
}
function axisTitle(){ return X==="t" ? "wall-clock minutes" : "simulated env steps"; }
function refresh(){
  for(const ch of charts){ ch.data.datasets = datasets(ch._g, ch._y, ch._p);
    ch.options.scales.x.title.text = axisTitle();
    ch.options.scales.x.type = X==="t" ? "linear" : "logarithmic"; ch.update(); }
}
mk("g-return","ep_return",{ylabel:"return"});
mk("g-surv","ep_survival",{ylabel:"fraction surviving",min:0,max:1});
mk("g-err","course_err_vel_xy",{ylabel:"m/s",min:0});
mk("g-train",null,{progress:true, ylabel:"center return"});
refresh();
document.querySelectorAll("#xaxis button").forEach(b=>b.onclick=()=>{
  X=b.dataset.x; document.querySelectorAll("#xaxis button").forEach(o=>o.setAttribute("aria-pressed", o===b)); refresh();});
// Final table
const t = document.getElementById("tbl");
const f = (v,d=2)=> (v===undefined||v===""||!isFinite(v)) ? "–" : Number(v).toFixed(d);
t.innerHTML = `<tr><th>Run</th><th>Params</th><th>Train min</th><th>Env steps</th><th>Return</th><th>Survival</th>
<th>Course falls</th><th>Err xy m/s</th><th>Err yaw rad/s</th><th>Time to 50</th></tr>` +
RUNS.map(r=>{ const e=r.eval[r.eval.length-1]; const hit=r.eval.find(x=>x.ep_return>=50);
 return `<tr><td><span class="sw" style="background:${color(r)}"></span>${r.label}</td>
 <td>${r.num_params? r.num_params.toLocaleString(): "–"}</td><td>${f(e.wall_time_s/60,0)}</td>
 <td>${(e.env_steps/1e6).toFixed(0)}M</td><td>${f(e.ep_return,1)}</td><td>${f(e.ep_survival)}</td>
 <td>${f(e.course_falls)}</td><td>${f(e.course_err_vel_xy)}</td><td>${f(e.course_err_vel_yaw)}</td>
 <td>${hit? f(hit.wall_time_s/60,0)+" min" : "never"}</td></tr>`;}).join("");
</script></body></html>
"""

if __name__ == "__main__":
    main()
