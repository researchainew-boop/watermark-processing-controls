"""Render the IEEE manuscript from completed measured runs and figure assets."""
from __future__ import annotations
import json
import math
import re
import csv
import itertools
from bisect import bisect_right
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "conference_clean"


def f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def ci(values: list[float], digits: int = 3) -> str:
    return ", ".join(f(v, digits) for v in values)


def roc_coordinates(rows: list[dict], column: str) -> str:
    """Empirical ROC, linearly drawn on a dense FPR grid for PGFPlots.

    The grid is for the illustration only. Reported metrics and hard-budget
    TPRs use every realizable score threshold in prepare_conference.py.
    """
    ordered = sorted([(float(row[column]), int(row["label"])) for row in rows], reverse=True)
    positives = sum(label for _, label in ordered)
    negatives = len(ordered) - positives
    xs, ys = [0.0], [0.0]
    tp = fp = 0
    for _, tied in itertools.groupby(ordered, key=lambda pair: pair[0]):
        for _, label in tied:
            tp += label
            fp += 1 - label
        xs.append(fp / negatives)
        ys.append(tp / positives)
    grid = sorted(set([i / 2000 for i in range(101)] + [i / 200 for i in range(10, 201)]))
    points = ["(0,0)"]
    for x in grid:
        i = bisect_right(xs, x) - 1
        y = ys[i]
        if i + 1 < len(xs) and xs[i + 1] > xs[i]:
            y += (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
        points.append(f"({x:.5f},{y:.5f})")
    return " ".join(points)


def roc_figure() -> str:
    with (OUT / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    removal = {("crop", 1), ("lowres", 2), ("regenerate", 2), ("spectral", 1)}
    slices = [rows, [row for row in rows if (row["attack"], int(row["severity"])) in removal]]
    scores = [
        ("probe_logit", "blue!75!black,solid", "ResNet probe"),
        ("fusion_logit", "orange!90!black,dashed", "ResNet fusion"),
        ("clip_probe_logit", "green!50!black,dashdotted", "CLIP probe"),
        ("clip_fusion_logit", "purple,densely dotted", "CLIP fusion"),
        ("active_score", "gray,dashed", "Payload score"),
    ]
    if any(column not in rows[0] for column, _, _ in scores):
        raise ValueError("Refresh decision-score statistics before rendering the ROC figure")
    lines = [r"\begin{figure*}[t]", r"\centering", r"\begin{tikzpicture}",
             r"\begin{groupplot}[group style={group size=2 by 1,horizontal sep=0.45in},width=0.44\textwidth,height=2.15in,xmin=0,xmax=1,ymin=0,ymax=1,xlabel={False-positive rate},xtick={0,0.25,0.5,0.75,1},ytick={0,0.25,0.5,0.75,1},tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\footnotesize},grid=major,grid style={gray!20}]" ]
    for i, subset in enumerate(slices):
        title = "(a) Full test mixture" if i == 0 else "(b) Four reduced-recovery settings"
        legend = r",legend style={font=\scriptsize,at={(0.98,0.04)},anchor=south east,draw=none,fill=white}" if i == 0 else ""
        ylabel = r",ylabel={True-positive rate}" if i == 0 else ""
        lines.append(r"\nextgroupplot[title={" + title + "}" + ylabel + legend + "]")
        for column, style, label in scores:
            lines.append(r"\addplot[" + style + r",line width=0.9pt,no marks] coordinates {" + roc_coordinates(subset, column) + "};")
            if i == 0:
                lines.append(r"\addlegendentry{" + label + "}")
        lines.append(r"\addplot[black,densely dotted,line width=0.4pt,forget plot] coordinates {(0.05,0)(0.05,1)};")
    lines.extend([r"\end{groupplot}", r"\end{tikzpicture}",
                  r"\caption{Origin ROC curves on the same retained test sources: (a) all " + f"{len(rows):,}" + r" variants and (b) crop, 32$\times$32 round trip, compound grid shift, and spectral wipe (" + f"{len(slices[1]):,}" + r" variants). Logistic curves rank linear decision scores; the payload curve ranks bit agreement. The dotted vertical line marks 5\% FPR. The second panel is a retrospective diagnostic slice, not an inference-time attack selector.}",
                  r"\label{fig:roc}", r"\end{figure*}"])
    return "\n".join(lines)


def main() -> None:
    r = json.loads((OUT / "metrics.json").read_text())
    if set(r["variants"]) != {"full", "no_attacks", "no_matched_reals"}:
        raise ValueError("All corrected variants must finish before the paper is generated")
    p = json.loads((OUT / "provenance.json").read_text())
    full = r["variants"]["full"]
    s = full["systems"]
    um = r["variants"]["no_matched_reals"]["systems"]
    na = r["variants"]["no_attacks"]["systems"]
    a = r["audit"]
    counts = a["sources_by_split_generator"]
    d = full["removal_slice"]
    gp = full["payload_by_group"]
    policy = full["decision"]
    valpol = full["decision_validation"]
    pg = policy["group_positive_rate"]
    head = full["attack_heads"]["attack_head"]
    matched = full["attack_heads_matched"]
    cleanclip = full["clean_unmarked"]["clip_probe"]
    source_rows = [" & ".join([label, *[str(counts.get(f"{split}|{gen}", 0)) for gen in ["real", "stable-diffusion", "dalle-3", "midjourney"]]]) + r"\\" for split, label in [("train", "Train"), ("val", "Validation"), ("test", "Test")]]
    main_rows = [" & ".join([label, f(m["roc_auc"]) + " [" + ci(m["roc_auc_ci"]) + "]", f(m["false_positive_rate"]), f(m["recall"]), f(m["tpr_at_fpr_5pct"])]) + r"\\" for key, label in [("active", "Payload"), ("probe", "ResNet probe"), ("fusion", "ResNet fusion"), ("clip_probe", "CLIP probe"), ("clip_fusion", "CLIP fusion")] for m in [s[key]]]
    attack_rows = []
    for key, label in [("jpeg:1", "JPEG 70"), ("jpeg:2", "JPEG 50"), ("blur:2", "Blur, kernel 5"), ("lowres:1", "Low-res 64"), ("crop:1", "Crop, 12 px/side"), ("lowres:2", r"Low-res 32$^\star$"), ("regenerate:2", r"Grid shift$^\star$"), ("spectral:1", r"Spectral wipe$^\star$")]:
        c = full["conditions"][key]
        attack_rows.append(" & ".join([label, f(c["bit_accuracy"]), *[f(c["systems"][n]["roc_auc"]) for n in ["active", "probe", "fusion"]]]) + r"\\")
    abl_rows = [" & ".join([label, f(sys["probe"]["roc_auc"]), f(sys["probe"]["tpr_at_fpr_5pct"]), f(sys["clip_probe"]["roc_auc"])]) + r"\\" for label, sys in [("Full", s), ("No attacks", na), ("No matched reals", um)]]
    v = {
        "SOURCES":f"{a['retained_sources']:,}", "EXCLUDED":a["excluded_sources"],
        "TEST_IMAGES":f"{s['probe']['n']:,}", "TEST_SOURCES":f"{s['probe']['sources']:,}",
        "RESNET_PROBE":f(s["probe"]["roc_auc"]), "RESNET_FUSION":f(s["fusion"]["roc_auc"]),
        "RESNET_GAIN":f(s["fusion_minus_probe"]["estimate"]), "RESNET_GAIN_CI":ci(s["fusion_minus_probe"]["ci95"]),
        "CLIP_PROBE":f(s["clip_probe"]["roc_auc"]), "CLIP_FUSION":f(s["clip_fusion"]["roc_auc"]),
        "CLIP_GAIN":f(s["clip_fusion_minus_clip_probe"]["estimate"],4), "CLIP_GAIN_CI":ci(s["clip_fusion_minus_clip_probe"]["ci95"],4),
        "FULL_TPR5":f(s["probe"]["tpr_at_fpr_5pct"]), "UNMATCHED_TPR5":f(um["probe"]["tpr_at_fpr_5pct"]),
        "UNMATCHED_AUC":f(um["probe"]["roc_auc"]), "UNMATCHED_CLIP":f(um["clip_probe"]["roc_auc"]), "NOATTACK_AUC":f(na["probe"]["roc_auc"]),
        "RESNET_TPR5_CI":ci(s["probe"]["tpr_at_fpr_5pct_ci"]), "CLIP_TPR5_CI":ci(s["clip_probe"]["tpr_at_fpr_5pct_ci"]),
        "UNMATCHED_TPR5_CI":ci(um["probe"]["tpr_at_fpr_5pct_ci"]),
        "PSNR":f(r["embedding_quality"]["mean_psnr"],1), "MARKED_SOURCES":r["embedding_quality"]["n"],
        "PAYLOAD_BITS":math.ceil(s["active"]["threshold"] *16), "ACTIVE_FPR":f(s["active"]["false_positive_rate"],4),
        "SOURCE_ROWS":"\n".join(source_rows), "MAIN_ROWS":"\n".join(main_rows), "ATTACK_ROWS":"\n".join(attack_rows), "ABLATION_ROWS":"\n".join(abl_rows),
        "DEAD_IMAGES":f"{d['probe']['n']:,}", "DEAD_PROBE":f(d["probe"]["roc_auc"]), "DEAD_FUSION":f(d["fusion"]["roc_auc"]),
        "DEAD_PROBE_CI":ci(d["probe"]["roc_auc_ci"]), "DEAD_FUSION_CI":ci(d["fusion"]["roc_auc_ci"]),
        "DEAD_GAIN":f(d["fusion_minus_probe"]["estimate"]), "DEAD_GAIN_CI":ci(d["fusion_minus_probe"]["ci95"]),
        "DEAD_CLIP_PROBE":f(d["clip_probe"]["roc_auc"]), "DEAD_CLIP_FUSION":f(d["clip_fusion"]["roc_auc"]),
        "DEAD_CLIP_GAIN":f(d["clip_fusion_minus_clip_probe"]["estimate"]), "DEAD_CLIP_GAIN_CI":ci(d["clip_fusion_minus_clip_probe"]["ci95"]),
        "ATTACK_AUC":f(head["roc_auc"]), "ATTACK_AUC_CI":ci(head["roc_auc_ci"]), "ATTACK_TPR5":f(head["tpr_at_fpr_5pct"]),
        "MATCHED_ATTACK_TPR5":f(matched["attack_head"]["tpr_at_fpr_5pct"]), "MATCHED_CLIP_ATTACK_TPR5":f(matched["clip_attack_head"]["tpr_at_fpr_5pct"]),
        "DECISION_TAU":f(policy["tau_active"],5), "DECISION_MARGIN":f(policy["margin"],2), "DECISION_BITS":math.ceil(policy["tau_active"]*16),
        "VAL_COVERAGE":f(valpol["coverage"]), "VAL_RT_FPR":f(valpol["group_positive_rate"]["R-T"]["positive_rate"]),
        "COVERAGE":f(policy["coverage"]), "DECIDED_BA":f(policy["balanced_accuracy_decided"]), "DECISION_RC":f(pg["R-C"]["positive_rate"]), "DECISION_RT":f(pg["R-T"]["positive_rate"]),
        "DECISION_SU_TPR":f(pg["S-U"]["positive_rate"]), "DECISION_SU_ABSTAIN":f(pg["S-U"]["abstain_rate"]), "DECISION_SU_FALSE_REAL":f(1-pg["S-U"]["positive_rate"]-pg["S-U"]["abstain_rate"]),
        "DEAD_COVERAGE":f(full["decision_removal_slice"]["coverage"]), "CLEAN_CLIP":f(cleanclip["roc_auc"]), "CLEAN_CLIP_CI":ci(cleanclip["roc_auc_ci"]),
        "BOOTSTRAPS":r["bootstrap_samples"], "PYTHON_VERSION":p["python"].split()[0], "SKLEARN_VERSION":p["packages"]["scikit-learn"], "NUMPY_VERSION":p["packages"]["numpy"],
    }
    for token, g in [("SW_ACCEPT","S-W"),("SA_ACCEPT","S-A"),("SU_ACCEPT","S-U"),("RC_ACCEPT","R-C"),("RT_ACCEPT","R-T")]:
        v[token] = f(gp[g]["accept_rate"],4 if g.startswith("R") else 3)
    mj = full["generator_midjourney"]
    for token, key in [("MJ_PROBE","probe"),("MJ_FUSION","fusion"),("MJ_CLIP","clip_probe")]:
        v[token] = f(mj[key]["roc_auc"])
    v["BIBLIOGRAPHY"] = (ROOT / "manuscript" / "conference_bibliography.txt").read_text(encoding="utf-8")
    v["ROC_FIGURE"] = roc_figure()
    from conference_extensions import values, order_bibliography
    v.update(values())
    from phase3_paper import values as phase3_values
    v.update(phase3_values(v))
    from phase4_paper import values as phase4_values
    v.update(phase4_values(v))
    if 'MATCH_R_GAIN' in (ROOT / 'manuscript/conference_template.tex').read_text(encoding='utf-8'):
        from phase5_paper import values as phase5_values
        v.update(phase5_values(v))
    template = (ROOT / "manuscript" / "conference_template.tex").read_text(encoding="utf-8")
    for key, value in v.items():
        if key == 'BIBLIOGRAPHY':
            continue
        template = template.replace("{{"+key+"}}", str(value))
    template = template.replace('{{BIBLIOGRAPHY}}',order_bibliography(v['BIBLIOGRAPHY'],template))
    unresolved = re.findall(r"\{\{[^}]+\}\}", template)
    if unresolved:
        raise ValueError(unresolved)
    (ROOT / "manuscript" / "main.tex").write_text(template, encoding="utf-8")
    (OUT / "manuscript_numbers.json").write_text(json.dumps({key:value for key,value in v.items() if key != "ROC_FIGURE"},indent=2),encoding="utf-8")
    print("Updated manuscript/main.tex from corrected measured results")


if __name__ == "__main__":
    main()
