# Closing the gunshot-visibility gap + fixing the stale README — implementation runbook

Every anchor, every code block, and every "expected result" below was
checked against the real repository before this plan was written,
including actually executing the dashboard's JavaScript under Node against
real data. Follow the tasks in strict order.

## HOW TO EXECUTE THIS PLAN

1. Work through tasks in order: T1 through T9.
2. **T1's fix requires re-running the 10-seed experiment (T2) to actually
   see the new numbers.** The patch alone doesn't retroactively add data to
   already-completed runs — it changes what gets captured on the *next*
   run. This is expected, not a sign something went wrong.
3. **Do not run `rm -rf artifacts` in this repository.**
   `artifacts/real_audio_validation/` is committed to git (it holds real
   experiment results, not regenerable build output) and a blanket
   `rm -rf artifacts` deletes it along with the disposable stuff. If you
   need a clean slate for the *main* pipeline's own outputs, delete
   specific files (`pipeline_report.json`, `dashboard.html`,
   `snn_model.npz`, etc.) instead of the whole directory. If you do delete
   it by accident, `git checkout HEAD -- artifacts/real_audio_validation`
   restores it losslessly, since it's already committed.
4. New files: none in this plan. Every task edits an existing file with an
   exact anchor, or is a re-run/verification step.
5. T2 (the actual 10-seed re-run) takes a long time — the original run
   used 10 seeds at 60 epochs each; budget accordingly and use the
   `setsid`-detached technique from prior plans, not a foreground command.

---

## TASK T1 — add per-class tracking to the iteration script

**The problem this fixes:** the currently committed multi-seed experiment
only saves one aggregate number
(`test_accuracy_real_only`, currently 65.8% ± 3.5%) covering gunshot,
vehicle, and ambient combined. Separately, the zero-shot experiment showed
gunshot detection failing almost completely on real audio (0% correctly
classified, only 9.1% of real gunshots triggering any alert at all,
committed in `experiment3_us8k_zero_shot.json`). **Whether retraining
fixed that specific, most operationally important failure is currently
unknown** — the aggregate accuracy could be hiding a persistent gunshot
problem behind decent vehicle/ambient performance, or gunshot could have
recovered along with everything else. `evaluate()` already computes the
per-class breakdown needed to answer this; the script just never saved it.

**Verified fact this fix depends on:** the training pipeline is fully
deterministic given a fixed seed. This was checked directly — running the
same seed through `build_corpus` → `build_spike_dataset` →
`a3.prepare_dataset` → `run_training_stage` twice produced byte-identical
accuracy both times. This means re-running with the same 10 seeds will
reproduce the exact same aggregate numbers already committed, while now
also capturing what was missing.

**Files touched:** `run_iterations_combined_datasets.py` only.

**Edit 1.** Find this exact text (inside `run_one_seed`):
```python
    model = stage2["model"]
    test_forests = np.asarray(prepared["forests"]["test"])
    real_mask = test_forests == "mixed"
    n_real = int(real_mask.sum())
    real_result = evaluate(model, prepared["test"]["spikes"][real_mask],
                            prepared["test"]["labels"][real_mask]) if n_real > 0 else None

    result = {
        "seed": seed,
        "n_synthetic": len(synthetic),
        "n_esc50": len(esc50_samples),
        "n_us8k": len(us8k_samples),
        "n_us8k_skipped": len(us8k_skipped),
        "test_accuracy_combined": stage2["test"]["accuracy"],
        "test_accuracy_real_only": real_result["accuracy"] if real_result else None,
        "n_real_in_test": n_real,
        "val_accuracy": stage2["val_accuracy"],
    }
```
Replace it with:
```python
    model = stage2["model"]
    test_forests = np.asarray(prepared["forests"]["test"])
    real_mask = test_forests == "mixed"
    n_real = int(real_mask.sum())
    real_result = evaluate(model, prepared["test"]["spikes"][real_mask],
                            prepared["test"]["labels"][real_mask]) if n_real > 0 else None

    # evaluate() already computes a per-class breakdown and confusion matrix
    # -- only "accuracy" was being kept before, which made it impossible to
    # tell whether gunshot specifically improved after retraining, or
    # whether a decent aggregate was hiding a persistent gunshot failure
    # behind good vehicle/ambient numbers. Save what was already being
    # computed instead of discarding it.
    gunshot_real_metrics = (
        real_result["per_class"].get("gunshot") if real_result and real_result.get("per_class") else None
    )

    result = {
        "seed": seed,
        "n_synthetic": len(synthetic),
        "n_esc50": len(esc50_samples),
        "n_us8k": len(us8k_samples),
        "n_us8k_skipped": len(us8k_skipped),
        "test_accuracy_combined": stage2["test"]["accuracy"],
        "test_accuracy_real_only": real_result["accuracy"] if real_result else None,
        "n_real_in_test": n_real,
        "val_accuracy": stage2["val_accuracy"],
        "test_confusion_matrix_real_only": real_result["confusion"].tolist() if real_result else None,
        "test_per_class_real_only": real_result.get("per_class") if real_result else None,
        "gunshot_real_recall": gunshot_real_metrics["recall"] if gunshot_real_metrics else None,
        "gunshot_real_precision": gunshot_real_metrics["precision"] if gunshot_real_metrics else None,
        "gunshot_real_support": gunshot_real_metrics["support"] if gunshot_real_metrics else None,
    }
```

**Edit 2.** Find this exact text (inside `main`):
```python
        real_accs = [r["test_accuracy_real_only"] for r in all_results if r["test_accuracy_real_only"] is not None]
        summary = {
            "seeds_completed": [r["seed"] for r in all_results],
            "n_seeds_completed": len(all_results),
            "n_seeds_requested": len(args.seeds),
            "per_seed_results": all_results,
            "real_only_accuracy_mean": statistics.mean(real_accs) if real_accs else None,
            "real_only_accuracy_stdev": statistics.stdev(real_accs) if len(real_accs) > 1 else None,
            "real_only_accuracy_min": min(real_accs) if real_accs else None,
            "real_only_accuracy_max": max(real_accs) if real_accs else None,
        }
```
Replace it with:
```python
        real_accs = [r["test_accuracy_real_only"] for r in all_results if r["test_accuracy_real_only"] is not None]
        # Aggregate gunshot recall specifically, the same way real_only
        # accuracy is aggregated -- this is the number that answers "did
        # retraining fix real gunshot detection" rather than only "did the
        # average of gunshot+vehicle+ambient improve." A seed contributes
        # to this aggregate only if at least one real gunshot clip actually
        # landed in that seed's held-out test split (gunshot_real_support
        # is not None and > 0) -- a seed where none did says nothing about
        # gunshot and must not silently count as "0% recall".
        gunshot_recalls = [
            r["gunshot_real_recall"] for r in all_results
            if r.get("gunshot_real_support") is not None and r["gunshot_real_support"] > 0
        ]
        summary = {
            "seeds_completed": [r["seed"] for r in all_results],
            "n_seeds_completed": len(all_results),
            "n_seeds_requested": len(args.seeds),
            "per_seed_results": all_results,
            "real_only_accuracy_mean": statistics.mean(real_accs) if real_accs else None,
            "real_only_accuracy_stdev": statistics.stdev(real_accs) if len(real_accs) > 1 else None,
            "real_only_accuracy_min": min(real_accs) if real_accs else None,
            "real_only_accuracy_max": max(real_accs) if real_accs else None,
            "gunshot_real_recall_mean": statistics.mean(gunshot_recalls) if gunshot_recalls else None,
            "gunshot_real_recall_stdev": statistics.stdev(gunshot_recalls) if len(gunshot_recalls) > 1 else None,
            "n_seeds_with_real_gunshot_in_test": len(gunshot_recalls),
        }
```

**Edit 3 — a genuine pre-existing bug, unrelated to gunshot tracking, found while testing this patch.** Running with exactly one seed crashes on the final summary print (`stdev` is `None` when there are fewer than 2 data points, and `None` can't be formatted with `.1%`). This doesn't affect the already-completed 10-seed run, but it will affect this task's own smoke-test step below if left unfixed. Find this exact text:
```python
    print("\n=== FINAL SUMMARY ===")
    print(f"seeds: {summary['seeds_completed']}")
    print(f"real_only_accuracy: mean={summary['real_only_accuracy_mean']:.1%}"
          f"  stdev={summary['real_only_accuracy_stdev']:.1%}"
          f"  range=[{summary['real_only_accuracy_min']:.1%}, {summary['real_only_accuracy_max']:.1%}]")
```
Replace it with:
```python
    print("\n=== FINAL SUMMARY ===")
    print(f"seeds: {summary['seeds_completed']}")
    stdev_display = f"{summary['real_only_accuracy_stdev']:.1%}" if summary['real_only_accuracy_stdev'] is not None else "n/a (need 2+ seeds)"
    print(f"real_only_accuracy: mean={summary['real_only_accuracy_mean']:.1%}"
          f"  stdev={stdev_display}"
          f"  range=[{summary['real_only_accuracy_min']:.1%}, {summary['real_only_accuracy_max']:.1%}]")
    if summary["gunshot_real_recall_mean"] is not None:
        gstdev = (f"{summary['gunshot_real_recall_stdev']:.1%}"
                  if summary["gunshot_real_recall_stdev"] is not None else "n/a (need 2+ qualifying seeds)")
        print(f"gunshot_real_recall: mean={summary['gunshot_real_recall_mean']:.1%}  stdev={gstdev}"
              f"  (n_seeds_with_real_gunshot_in_test={summary['n_seeds_with_real_gunshot_in_test']}/{summary['n_seeds_completed']})")
    else:
        print("gunshot_real_recall: no seed had a real gunshot clip in its held-out test split -- "
              "check max_per_us8k_category and n_frames/seed settings")
```

**Smoke-test verification (small settings, confirms no crash, not a real
result):**
```
python3 run_iterations_combined_datasets.py \
  --esc50-root /path/to/ESC-50 --us8k-root /path/to/UrbanSound8K \
  --seeds 901 902 --epochs 3 --n-synthetic 40 --max-per-us8k-category 5 --n-frames 100 \
  --out-dir /tmp/gunshot_patch_smoke
```
**Expected result:** completes without a traceback, and the final summary
now includes a `gunshot_real_recall:` line in addition to the existing
`real_only_accuracy:` line. This exact command (against ESC-50 and a small
mock UrbanSound8K standing in for the real dataset) produced:
```
real_only_accuracy: mean=58.9%  stdev=6.2%  range=[54.5%, 63.3%]
gunshot_real_recall: mean=0.0%  stdev=0.0%  (n_seeds_with_real_gunshot_in_test=2/2)
```
Do not expect these exact percentages against your real data — the mock
only had one tiny gunshot clip — but the *shape* of the output (both lines
present, no crash) must match.

**Also verify resumability still works** (a feature already in the
committed script, not part of this patch, that must survive it):
```
python3 run_iterations_combined_datasets.py \
  --esc50-root /path/to/ESC-50 --us8k-root /path/to/UrbanSound8K \
  --seeds 901 902 903 --epochs 3 --n-synthetic 40 --max-per-us8k-category 5 --n-frames 100 \
  --out-dir /tmp/gunshot_patch_smoke
```
**Expected result:** the log shows `seed 901 (already completed,
skipping)` and `seed 902 (already completed, skipping)`, then only seed
903 actually runs, and the final summary now covers all three.

---

## TASK T2 — re-run the real 10-seed experiment

**Preconditions:** T1 verified.

**This produces the actual answer to "did retraining fix real gunshot
detection."** Use the same seeds as the original committed run so the
aggregate numbers act as a cross-check (they should match the already
-committed 65.8% ± 3.5% almost exactly, per T1's determinism note):

```
setsid nohup python3 run_iterations_combined_datasets.py \
  --esc50-root /path/to/ESC-50 --us8k-root /path/to/UrbanSound8K \
  --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 60 --max-per-us8k-category 60 \
  --out-dir artifacts/real_audio_validation/iterations \
  > /tmp/gunshot_rerun.log 2>&1 < /dev/null &
sleep 5
ps aux | grep run_iterations | grep -v grep
```
**Expected result:** one line showing the process running.

**Poll every few minutes, as separate commands:**
```
tail -15 /tmp/gunshot_rerun.log
ps aux | grep run_iterations | grep -v grep
```

**Once finished, verify:**
```
python3 -c "
import json
s = json.load(open('artifacts/real_audio_validation/iterations/summary.json'))
print('real_only_accuracy_mean:', s['real_only_accuracy_mean'])
print('gunshot_real_recall_mean:', s['gunshot_real_recall_mean'])
print('gunshot_real_recall_stdev:', s['gunshot_real_recall_stdev'])
print('n_seeds_with_real_gunshot_in_test:', s['n_seeds_with_real_gunshot_in_test'])
assert s['n_seeds_with_real_gunshot_in_test'] > 0, 'no seed had any real gunshot in its test split -- check settings'
"
```
**Expected result:** `real_only_accuracy_mean` close to `0.6578...` (the
already-committed value) confirming determinism held; `gunshot_real_recall_mean`
is the number this whole exercise was for — there is no
predetermined "correct" value to expect, since this has never been
measured before. Whatever it is, that is the finding. Do not adjust
`max_per_us8k_category` or seeds after seeing an unwelcome number to try
to get a better one — report what this run says.

---

## TASK T3 — add per-class real-audio results to the dashboard's data source

**Preconditions:** none (independent of T1/T2, but more useful after them).

**Files touched:** `ecosentry/pipeline.py` only.

Find this exact text:
```python
def render_dashboard(report: Dict, out_dir: Path) -> Path:
    """Fill dashboard_template.html with this run's pipeline_report.json
    and write the result to out_dir/dashboard.html (Phase 0, Task P0.3)."""
    template_path = Path(__file__).parent / "dashboard_template.html"
    template = template_path.read_text(encoding="utf-8")
    injected = template.replace("__REPORT_JSON__", json.dumps(report, default=str))
    out_path = out_dir / "dashboard.html"
    out_path.write_text(injected, encoding="utf-8")
    return out_path
```
Replace it with:
```python
def _load_real_audio_validation(out_dir: Path) -> Optional[Dict]:
    """Best-effort load of committed real-audio validation results, if any
    exist under out_dir/real_audio_validation/. Returns None if the
    directory or its expected files are missing -- this must never raise,
    since real-audio validation runs separately (it needs external
    datasets not bundled with this repo) and is optional from the main
    pipeline's point of view."""
    base = out_dir / "real_audio_validation"
    result: Dict = {}
    for key, filename in (
        ("esc50_zero_shot", "experiment1_zero_shot.json"),
        ("esc50_retrain", "experiment2_retrain.json"),
        ("us8k_zero_shot", "experiment3_us8k_zero_shot.json"),
        ("iterations_summary", "iterations/summary.json"),
    ):
        path = base / filename
        if path.exists():
            try:
                result[key] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return result or None


def render_dashboard(report: Dict, out_dir: Path) -> Path:
    """Fill dashboard_template.html with this run's pipeline_report.json
    and write the result to out_dir/dashboard.html (Phase 0, Task P0.3).
    Also attaches committed real-audio validation results, if present,
    under report["real_audio_validation"] -- see _load_real_audio_validation."""
    template_path = Path(__file__).parent / "dashboard_template.html"
    template = template_path.read_text(encoding="utf-8")
    report = dict(report)
    report["real_audio_validation"] = _load_real_audio_validation(out_dir)
    injected = template.replace("__REPORT_JSON__", json.dumps(report, default=str))
    out_path = out_dir / "dashboard.html"
    out_path.write_text(injected, encoding="utf-8")
    return out_path
```
`Optional` is already imported at the top of `pipeline.py`
(`from typing import Dict, List, Optional, Sequence`) — do not add a
duplicate import.

**Verification:**
```
python3 -c "
from ecosentry.pipeline import _load_real_audio_validation
from pathlib import Path
import tempfile

data = _load_real_audio_validation(Path('artifacts'))
assert data is not None
assert 'esc50_zero_shot' in data and 'us8k_zero_shot' in data and 'iterations_summary' in data
print('OK -- found committed real-audio validation data:', sorted(data.keys()))

with tempfile.TemporaryDirectory() as d:
    assert _load_real_audio_validation(Path(d)) is None
print('OK -- missing directory returns None, no crash')
"
```
**Expected result:**
```
OK -- found committed real-audio validation data: ['esc50_retrain', 'esc50_zero_shot', 'iterations_summary', 'us8k_zero_shot']
OK -- missing directory returns None, no crash
```

---

## TASK T4 — the mandatory existing-test fix this causes

**Why this is required, not optional:** `render_dashboard` now always adds
a `real_audio_validation` key to whatever report dict it's given, even
when that key wasn't in the caller's original input. One existing test
assumed byte-for-byte pass-through and will fail after T3 — this is a
deliberate, correct consequence of T3's change, not a sign T3 broke
something.

**Files touched:** `tests/test_dashboard_rendering.py` only. This is the
one pre-existing test this plan edits — every other test change in this
plan is new, not a modification to something that already existed.

Find this exact text:
```python
def test_render_dashboard_embeds_report_json(tmp_path):
    report = {"hello": "world", "n": 3}
    out_path = render_dashboard(report, tmp_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "__REPORT_JSON__" not in content
    assert json.dumps(report) in content
```
Replace it with:
```python
def test_render_dashboard_embeds_report_json(tmp_path):
    # render_dashboard now attaches report["real_audio_validation"]
    # (None, in tmp_path, since no real_audio_validation/ subdirectory
    # exists there) alongside whatever the caller passed in -- so the
    # embedded JSON is the caller's report PLUS that one extra key, not
    # byte-identical to the caller's report on its own. This is a
    # deliberate behavior change (see ecosentry/pipeline.py's
    # render_dashboard docstring), not a regression -- this test checks
    # the original keys survive and the new key is present, rather than
    # requiring exact equality with the input.
    report = {"hello": "world", "n": 3}
    out_path = render_dashboard(report, tmp_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "__REPORT_JSON__" not in content
    assert '"hello": "world"' in content
    assert '"n": 3' in content
    assert '"real_audio_validation": null' in content
```

**Verification:**
```
python -m pytest tests/test_dashboard_rendering.py -v
```
**Expected result:** `2 passed`.

**Full-suite regression check:**
```
python -m pytest tests/ -q
```
**Expected result:** `145 passed`.

---

## TASK T5 — add the dashboard section markup

**Preconditions:** none.

**Files touched:** `ecosentry/dashboard_template.html` only.

**Edit 1 — the new section, plus a stale test-count fix found while
locating the anchor.** Find this exact text:
```html
  <section id="delivery-visualizer">
    <div class="section-head">
      <h2>Priority delivery pipeline</h2>
      <div class="note">End-to-end latency and delivery funnel for the priority alert path, from this run's delivery_report.json</div>
    </div>
    <div id="delivery-funnel"></div>
  </section>

</main>
```
Replace it with:
```html
  <section id="delivery-visualizer">
    <div class="section-head">
      <h2>Priority delivery pipeline</h2>
      <div class="note">End-to-end latency and delivery funnel for the priority alert path, from this run's delivery_report.json</div>
    </div>
    <div id="delivery-funnel"></div>
  </section>

  <section id="real-audio-validation">
    <div class="section-head">
      <h2>Real-audio validation</h2>
      <div class="note">Zero-shot transfer and retraining results against ESC-50 and UrbanSound8K -- see artifacts/real_audio_validation/SUMMARY.md and SUMMARY_US8K.md for full detail</div>
    </div>
    <div id="real-audio-content"></div>
  </section>

</main>
```

**Edit 2 — separately, fix a stale count noticed in the footer while
working in this file** (unrelated to the section above, but in the same
file and trivial to fix alongside it): find this exact text:
```
      110 tests &middot; ~20s<br/>
```
Replace it with:
```
      145 tests &middot; ~20s<br/>
```

**Verification:**
```
grep -c 'id="real-audio-validation"' ecosentry/dashboard_template.html
grep -c '145 tests' ecosentry/dashboard_template.html
```
**Expected result:** `1` and `1`.

---

## TASK T6 — add the dashboard rendering script

**Preconditions:** T5 complete.

**Files touched:** `ecosentry/dashboard_template.html` only.

Find this exact text (the end of the existing bootstrap script):
```javascript
    svg += '</svg>';
    funnelContainer.innerHTML = svg;
  }
})();
</script>
```
Replace it with:
```javascript
    svg += '</svg>';
    funnelContainer.innerHTML = svg;
  }

  var rav = report.real_audio_validation;
  var ravContainer = document.getElementById('real-audio-content');
  if (ravContainer) {
    if (!rav) {
      ravContainer.innerHTML = '<p class="note">Not yet run in this checkout. '
        + 'See ESC50_REAL_AUDIO_VALIDATION_PLAN.md and URBANSOUND8K_INTEGRATION_PLAN.md '
        + 'to generate artifacts/real_audio_validation/.</p>';
    } else {
      var rows = '';
      function pct(x) { return (x === null || x === undefined) ? 'n/a' : (x * 100).toFixed(1) + '%'; }

      if (rav.esc50_zero_shot) {
        rows += '<tr><td>ESC-50 zero-shot (argmax accuracy)</td><td>' + pct(rav.esc50_zero_shot.argmax_accuracy)
          + '</td><td>n=' + rav.esc50_zero_shot.n_real_clips + '</td></tr>';
      }
      if (rav.esc50_retrain) {
        rows += '<tr><td>ESC-50 retrained (real-only accuracy)</td><td>' + pct(rav.esc50_retrain.test_accuracy_real_only)
          + '</td><td>n=' + rav.esc50_retrain.n_real_in_test_split + '</td></tr>';
      }
      if (rav.us8k_zero_shot && rav.us8k_zero_shot.alert_decision_by_true_class
          && rav.us8k_zero_shot.alert_decision_by_true_class.gunshot) {
        var g = rav.us8k_zero_shot.alert_decision_by_true_class.gunshot;
        rows += '<tr><td>UrbanSound8K zero-shot gunshot alert rate</td><td>' + pct(g.alert_rate)
          + '</td><td>n=' + g.n + '</td></tr>';
      }
      if (rav.iterations_summary) {
        var it = rav.iterations_summary;
        rows += '<tr><td>Combined multi-seed real-only accuracy</td><td>' + pct(it.real_only_accuracy_mean)
          + (it.real_only_accuracy_stdev != null ? ' &plusmn; ' + pct(it.real_only_accuracy_stdev) : '')
          + '</td><td>' + it.n_seeds_completed + ' seeds</td></tr>';
        if ('gunshot_real_recall_mean' in it) {
          rows += '<tr><td>Combined multi-seed gunshot recall (real)</td><td>' + pct(it.gunshot_real_recall_mean)
            + (it.gunshot_real_recall_stdev != null ? ' &plusmn; ' + pct(it.gunshot_real_recall_stdev) : '')
            + '</td><td>' + (it.n_seeds_with_real_gunshot_in_test || 0) + ' qualifying seeds</td></tr>';
        } else {
          rows += '<tr><td>Combined multi-seed gunshot recall (real)</td><td colspan="2">'
            + 'not tracked in this run -- re-run after applying the per-class tracking patch</td></tr>';
        }
      }
      ravContainer.innerHTML = '<table class="rav-table"><thead><tr><th>Metric</th><th>Value</th><th>Sample size</th></tr></thead><tbody>'
        + rows + '</tbody></table>';
    }
  }
})();
</script>
```

This code deliberately handles the case where T1/T2 haven't been run yet
(`gunshot_real_recall_mean` absent from `iterations_summary`) by showing an
explanatory placeholder instead of `undefined` or a blank cell — so this
task can be done independently of T1/T2 and will automatically pick up the
richer data once T2 completes, with no further template changes needed.

**Verification — this exact JS was extracted and executed under Node
against the real committed data before this plan was written:**
```
python3 -c "
from ecosentry.pipeline import render_dashboard
from pathlib import Path
import json
report = {'simulations': {'energy': {}, 'network': {}}, 'delivery': {}}
render_dashboard(report, Path('artifacts'))
print('rendered OK')
"
python -m ecosentry run --preset quick
```
Then open `artifacts/dashboard.html` in a browser and confirm a "Real-audio
validation" section appears near the bottom with a 4-row table. On the
verified run (before T2's re-run), that table read:
```
ESC-50 zero-shot (argmax accuracy)          15.6%          n=480
ESC-50 retrained (real-only accuracy)       67.8%          n=146
UrbanSound8K zero-shot gunshot alert rate   9.1%           n=374
Combined multi-seed real-only accuracy      65.8% ± 3.5%   10 seeds
Combined multi-seed gunshot recall (real)   not tracked in this run -- re-run after applying the per-class tracking patch
```
After T2 completes, the last row should instead show a real percentage
with a sample size, not the placeholder text.

---

## TASK T7 — fix the stale "ambient rejection: not fixed" section

**Preconditions:** none. This section describes a problem that was fixed
several commits ago (`0159167`) but the README was never updated —
independently verified just now by re-running the full pipeline, which
prints `[PASS] ambient_false_alert_rate 0.0%` in its acceptance table.

**Files touched:** `README.md` only.

Find this exact text:
```markdown
### One result that is not good: ambient rejection

**47.5% of pure-ambient clips produced a false alert.** Of 40 forest-background
clips containing no threat at all, 19 crossed the alert threshold.

This is a design gap, not a bug. `ARCH_4`/`ARCH_5` specify a **3-class** head —
gunshot, chainsaw, vehicle — with a softmax over exactly those three. A softmax
has no "none of these" option, so a clip of birdsong is forced into one of the
three, and roughly half the time it lands above the confidence threshold. The
real device listens to ambient forest ~99.99% of the time, so this is the
metric that would dominate field performance.

`ARCH_3`'s own class map already defines `3 = ambient`; the training head does
not use it. The fix is one of:

1. **Train a 4-class head** including ambient (this codebase already carries
   ambient clips through ARCH_1–3 and holds them out as negatives — set
   `SNNConfig.n_classes = 4` and include class 3 in `pipeline.TRAIN_CLASSES`).
2. **Add an energy/novelty gate** before the SNN so silent windows never reach
   the classifier — which also saves the inference energy.
3. **Raise the thresholds** from a real precision-recall curve, accepting lower
   recall.

Option 1 is the smallest change and is the one the documents implicitly point
at. It is deliberately left undone here rather than silently changing the
specified architecture.
```
Replace it with:
```markdown
### Ambient rejection: fixed (was a 47.5% false-alert rate)

The 3-class head originally shipped here — gunshot, chainsaw, vehicle, no
"none of these" option — forced every ambient clip into one of three
threats, producing a **47.5% false-alert rate** on threat-free audio.
`ARCH_3`'s own class map already defined `3 = ambient`; the training head
just didn't use it.

Fixed by training a 4-class head (`SNNConfig.n_classes = 4`,
`pipeline.TRAIN_CLASSES` including class 3) and rewiring the false-alert
measurement to read from the held-out test split instead of a separate
negative set. Re-verified against a fresh `--preset full` run: **0.0%
ambient false-alert rate**, both on the held-out test split (n=11) and on
freshly synthesized ambient audio the model never saw in any split during
training (n=11, out-of-distribution check). See `ARCH_7_ENERGY_PROFILER.md`'s
correction log and `tests/test_ambient_rejection.py` for the fix and its
regression tests.
```

**Verification:**
```
grep -n "3-class.*ambient\|Ambient rejection: fixed" README.md
```
**Expected result:** the new heading text appears; the old "47.5%... This
is a design gap" prose no longer does.

---

## TASK T8 — fix the remaining stale numbers and TODOs

**Preconditions:** none, but do this after T7 for a coherent diff.

**Files touched:** `README.md` only. Five separate edits in this file.

**Edit 1 — the acceptance table.** Find this exact text:
```markdown
### Acceptance table — 7 of 8 targets met

| Criterion | Measured | Target | |
|---|---|---|---|
| SNN test accuracy | **97.2%** | >85% | PASS |
| Spike firing rate | 25.0% normalised | 25% ±5% | PASS |
| Alert payload size | 116 B | <1000 B | PASS |
| End-to-end latency p95 | 630 ms | <1500 ms | PASS |
| Network delivery (worst forest) | 100% | >95% | PASS |
| Network latency p99 (worst forest) | 582 ms | <1500 ms | PASS |
| Operational endurance (worst forest) | 30 days | ≥30 days | PASS |
| Power reduction | 12× system / 18× compute | 50–75× | **FAIL** — see correction 8 |

Training reached 98.2% validation accuracy at epoch 118 in 359 s. Confusion
matrix on the 109-sample test split: gunshot 51/51, chainsaw 28/31,
vehicle 27/27.
```
Replace it with:
```markdown
### Acceptance table — 8 of 9 targets met

| Criterion | Measured | Target | |
|---|---|---|---|
| SNN test accuracy | **95.8%** | >85% | PASS |
| Spike firing rate | 25.0% normalised | 25% ±5% | PASS |
| Alert payload size | 116 B | <1000 B | PASS |
| End-to-end latency p95 | 825 ms | <1500 ms | PASS |
| Ambient false-alert rate | 0.0% | <10% | PASS |
| Network delivery (worst forest) | 100% | >95% | PASS |
| Network latency p99 (worst forest) | 582 ms | <1500 ms | PASS |
| Operational endurance (worst forest) | 30 days | ≥30 days | PASS |
| Power reduction | 12× system / 18× compute | 50–75× | **FAIL** — see correction 8 |

Training reached 95.8% test accuracy in a `--preset full` run (498.4 s wall
clock end to end). Confusion matrix on the 120-sample test split (now
4-class, including ambient): gunshot 51/51, chainsaw 29/31, vehicle 27/27,
ambient 8/11.
```
(The exact accuracy, latency, and wall-clock figures above are from one
verified run of `--preset full` and will vary slightly run to run — if
your own run produces materially different numbers, use yours instead of
copying these verbatim; the acceptance table should always reflect an
actual run, which is the whole point of this fix.)

**Edit 2 — correction item 10.** Find this exact text:
```markdown
10. **A 3-class softmax cannot express "nothing is happening".** Measured
    consequence: a 47.5% false-alert rate on threat-free audio. *Not fixed* —
    it changes the specified architecture, so it is written up above as the
    top open item rather than silently altered.
```
Replace it with:
```markdown
10. **A 3-class softmax cannot express "nothing is happening".** Measured
    consequence: a 47.5% false-alert rate on threat-free audio. *Fixed* —
    trained a 4-class head instead (see "Ambient rejection: fixed" above);
    re-measured at 0.0% on both held-out and out-of-distribution ambient audio.
```

**Edit 3 — the test count in the Testing section.** Find this exact text:
```markdown
110 tests, ~20 seconds. Coverage includes:
```
Replace it with:
```markdown
145 tests, ~20 seconds. Coverage includes:
```

**Edit 4 — the repository layout table's stale test count.** Find this
exact text:
```markdown
| `tests/` | — | 142 tests covering unit, protocol, energy, network, delivery & dashboard |
```
Replace it with:
```markdown
| `tests/` | — | 145 tests covering unit, protocol, energy, network, delivery, dashboard & real-audio validation |
```

**Edit 5 — add the new top-level scripts to the repository layout table.**
Find this exact text:
```markdown
| `tests/` | — | 145 tests covering unit, protocol, energy, network, delivery, dashboard & real-audio validation |

---

## Commands
```
Replace it with:
```markdown
| `tests/` | — | 145 tests covering unit, protocol, energy, network, delivery, dashboard & real-audio validation |
| `esc50_loader.py` | — | Loads ESC-50 clips mapped to the 4-class taxonomy (no gunshot -- ESC-50 has none) |
| `urbansound8k_loader.py` | — | Loads UrbanSound8K clips mapped to the 4-class taxonomy (fills the gunshot gap) |
| `run_experiment1_zero_shot.py` | — | Zero-shot: synthetic-trained model evaluated on real ESC-50 audio |
| `run_experiment3_us8k_zero_shot.py` | — | Zero-shot: synthetic-trained model evaluated on real UrbanSound8K audio (incl. gunshot) |
| `run_iterations_combined_datasets.py` | — | Multi-seed retraining on synthetic + ESC-50 + UrbanSound8K combined |

---

## Commands
```
(This edit's anchor deliberately includes the `---` and `## Commands`
lines from Edit 4's replacement so the two edits compose correctly if
applied in order — do not apply Edit 5 before Edit 4, since Edit 5's
anchor text won't exist yet.)

**Edit 6 — the stale "Requires real data" TODO section.** Find this exact
text:
```markdown
### Requires real data (highest priority for accuracy claims)

The corpus is **synthetic**. `ecosentry/synth.py` generates physically plausible
gunshots, chainsaws, vehicles and per-forest backgrounds, but the 97.2% the
pipeline reports is *accuracy on synthetic audio*. Synthetic classes are cleanly
separable by construction, so treat that figure as "the pipeline trains
correctly", not "the system detects poachers". Before any deployment claim:

1. Obtain ESC-50, UrbanSound8K and the three sets of field recordings the
   documents describe (600 clips).
2. Replace `build_corpus` with a loader for them.
3. Retrain and re-measure. Expect the accuracy to drop and the LIF gain and
   forest-normalisation parameters to need retuning.
```
Replace it with:
```markdown
### Real data: done for ESC-50 and UrbanSound8K; field recordings still open

The corpus used for the numbers above is still **synthetic**
(`ecosentry/synth.py`), and the 95.8% accuracy in the acceptance table is
still *accuracy on synthetic audio* — that figure means "the pipeline
trains correctly," not "the system detects poachers." Real-audio
validation now exists as a separate, explicit experiment track:

- **ESC-50** (no gunshot data available in this dataset): zero-shot
  transfer from the synthetic-trained model scored 15.6% accuracy;
  retraining with ESC-50 mixed into the corpus raised that to 67.8%
  (n=146, 5 seeds). See `ESC50_REAL_AUDIO_VALIDATION_PLAN.md` and
  `artifacts/real_audio_validation/SUMMARY.md`.
- **UrbanSound8K** (fills the gunshot gap ESC-50 can't): zero-shot gunshot
  detection on real audio was 9.1% alert rate, 0% correctly classified
  (n=374) — the model trained purely on synthetic audio essentially does
  not recognize a real gunshot. Combined multi-seed retraining (synthetic
  + ESC-50 + UrbanSound8K) reached 65.8% ± 3.5% real-only accuracy across
  10 seeds. See `URBANSOUND8K_INTEGRATION_PLAN.md` and
  `artifacts/real_audio_validation/SUMMARY_US8K.md`.
- Whether retraining specifically fixed real gunshot detection (as opposed
  to the aggregate improving via vehicle/ambient) is tracked in the
  `gunshot_real_recall` fields of `artifacts/real_audio_validation/
  iterations/summary.json` — see the "Real-audio validation" section of
  `dashboard.html` for the current answer.
- **Field recordings from an actual forest** (as opposed to ESC-50/
  UrbanSound8K's general-purpose and urban audio respectively) remain
  untested. `chainsaw` specifically has zero real-audio coverage from
  either dataset — neither contains a chainsaw-like category.
```

**Verification (all six edits at once):**
```
grep -c "97.2%" README.md
grep -c "8 of 9 targets met" README.md
grep -c "145 tests" README.md
grep -c "esc50_loader.py" README.md
grep -n "Not fixed" README.md
grep -c "Real data: done for ESC-50" README.md
```
**Expected result:** `0`, `1`, `2` (one in the Testing section, one in the
repository layout table row description — recount if your edits produced
a different number and check which occurrence is missing), `1` (in the
new repository layout row), no output at all from the `grep -n "Not
fixed"` line (it should find nothing, since that phrase no longer exists
anywhere), and `1`.

---

## TASK T9 — final full verification

**Preconditions:** all of T1–T8 complete (T2 may still be running in the
background if it's a long combined run — this final check can be done
either way, since T2 only affects data quality, not whether anything
crashes).

```
python -m pytest tests/ -q
python -m ecosentry run --preset quick
python3 -c "
import json
r = json.load(open('artifacts/pipeline_report.json'))
assert 'delivery' in r
print('pipeline still produces a full report: OK')
"
grep -c 'real-audio-validation' ecosentry/dashboard_template.html
grep -c '97.2%' README.md
```
**Expected result:**
```
145 passed
... (pipeline run output) ...
pipeline still produces a full report: OK
1
0
```
If the last line is anything other than `0`, a stale `97.2%` reference
survived T8 somewhere this plan's `grep` checks didn't catch — search for
it directly (`grep -n "97.2%" README.md`) and fix it using the same
principle as the rest of this plan: replace it with the number an actual
run produces, not a number carried over from before the ambient-rejection
fix.
