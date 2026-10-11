"""Render Phase5 manuscript numbers only from completed, locked measurements."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def rows(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def f(value, places=3):
    return f"{float(value):.{places}f}"


def interval(row, low="ci95_low", high="ci95_high", places=4):
    return f(row[low], places) + ", " + f(row[high], places)


def mean_sd(row, mean, sd):
    return "$" + f(row[mean]) + r"\pm" + f(row[sd]) + "$"


def values(old):
    for name in ["calibration/complete.json", "calibration/sd_family_excluded/complete.json",
                 "matching_factorial/complete.json", "paired_content/complete.json", "fair_watermarks/summary.json"]:
        if not (ROOT / "outputs/phase5" / name).is_file():
            raise RuntimeError(f"Cannot render pending scientific evidence: {name}")
    matching = rows("outputs/phase5/matching_factorial/mean_std_metrics.csv")
    mi = {(r["backbone"], r["cell"]): r for r in matching if r["scope"] == "overall"}
    contrast = rows("outputs/phase5/matching_factorial/paired_source_bootstrap.csv")
    ci = {(r["backbone"], r["metric"]): r for r in contrast if r["seed"] == "five_seed_mean"}
    result = {}
    for backbone, prefix in [("resnet", "R"), ("clip", "C")]:
        result[f"MATCH_{prefix}_GAIN"] = f(ci[backbone, "auc"]["estimate"], 4)
        result[f"MATCH_{prefix}_CI"] = interval(ci[backbone, "auc"])
        result[f"MATCH_{prefix}_TPR"] = f(ci[backbone, "tpr_at_validation_fpr5"]["estimate"], 4)
    result["FACTORIAL_ROWS"] = "\n".join(" & ".join([cell[0] + "/" + cell[1],
        mean_sd(mi["resnet", cell], "auc_mean", "auc_std"),
        mean_sd(mi["clip", cell], "auc_mean", "auc_std")]) + r"\\" for cell in ["AA", "AB", "BA", "BB"])
    unmarked = read("outputs/phase3/controls/unmarked_frozen_results.json")["expanded_test_overall"]
    for backbone, token in [("resnet", "EXPANDED_RESNET_AUCS"), ("clip", "EXPANDED_CLIP_AUCS")]:
        result[token] = "/".join(f(unmarked[f"{backbone}_{s}"]["auc"]) for s in ["probe", "fusion", "gated"])

    cal = rows("outputs/phase5/calibration/mean_std_ci.csv")
    index = {(r["backbone"], r["policy"], r["quantity"]): r for r in cal if
             r["dataset"] == "original" and r["processing_scope"] == "all16" and r["group"] == "overall" and r["system"] == "probe"}
    cal_rows = []
    for backbone, label in [("resnet", "ResNet"), ("clip", "CLIP")]:
        for policy, short in [("per_image_core13", "I"), ("conformal_core13", "S13"), ("conformal_all16", "S16")]:
            cal_rows.append(" & ".join([label, short, *[mean_sd(index[backbone, policy, q], "mean", "std")
                            for q in ["copy_TPR", "copy_FPR", "source_any_FPR"]]]) + r"\\")
    result["CALIBRATION_ROWS"] = "\n".join(cal_rows)
    def rate(backbone, policy, quantity):
        return f(index[backbone, policy, quantity]["mean"])
    result["PHASE5_CALIBRATION_RESULTS"] = r"""Table~\ref{tab:calibration} shows the recall cost of source-max calibration
on the sixteen-condition test family.
Some realized source FPRs still exceed 0.05 (maximum 0.066);
CLIP S16 TPR split SD is 0.132. Source-max core13 calibration does not cover
the added weak views: CLIP source FPR rises to 0.111 on all16 versus 0.019
on core13. These measured costs and variability qualify the marginal claim.
"""

    paired = read("outputs/phase5/paired_content/summary.json")
    assert paired["pairs"] == 200 and paired["copies"] == 1200 and not paired["test_refitting"] and not paired["test_threshold_tuning"]
    pi = {(r["regime"], r["backbone"], r["system"], r["condition"], r["policy"]): r for r in paired["all_metrics"]}
    def p(regime, backbone, condition="identity", policy="per_image_core13"):
        return pi[regime, backbone, "probe", condition, policy]
    result["PAIRED_R_AUC"] = f(p("joint", "resnet")["AUC"])
    result["PAIRED_C_AUC"] = f(p("joint", "clip")["AUC"])
    pair_rows = []
    for backbone, label in [("resnet", "ResNet"), ("clip", "CLIP")]:
        for regime, short in [("joint", "Joint"), ("SD_excluded", "SD-out")]:
            image = p(regime, backbone)
            source = p(regime, backbone, policy="conformal_all16")
            pair_rows.append(" & ".join([label, short, f(image["AUC"]), f(image["TPR"]), f(image["FPR"]), f(source["TPR"]), f(source["FPR"])]) + r"\\")
    result["PAIRED_CONTENT_ROWS"] = "\n".join(pair_rows)
    pair_text = r"""On the new caption-paired identity test, joint ResNet/CLIP AUC is
""" + f(p("joint", "resnet")["AUC"]) + "/" + f(p("joint", "clip")["AUC"]) + "; paired-bootstrap intervals are [" + interval(p("joint", "resnet"), "AUC_ci95_low", "AUC_ci95_high", 3) + "]/[" + interval(p("joint", "clip"), "AUC_ci95_low", "AUC_ci95_high", 3) + r"""].
Joint CLIP JPEG50/low-res32 AUC is """ + f(p("joint", "clip", "jpeg")["AUC"]) + "/" + f(p("joint", "clip", "lowres")["AUC"]) + r"""; its fixed image FPR is
""" + f(p("joint", "clip", "jpeg")["FPR"]) + "/" + f(p("joint", "clip", "lowres")["FPR"]) + r""". All heads and cuts remain fixed.
The SD-out arm (Table~\ref{tab:paired}) changes the synthetic training mix as well as supervised
generator exposure; it is a restricted transfer control.
"""
    pair_text += "On these unmarked identity images, ResNet fusion/gate AUC is " + f(pi["joint", "resnet", "fusion", "identity", "per_image_core13"]["AUC"]) + "/" + f(pi["joint", "resnet", "gated", "identity", "per_image_core13"]["AUC"]) + r" (Fig.~\ref{fig:roc})." + "\n"
    sensitivity = read("outputs/phase5/paired_content/postgeneration_overlap_sensitivity.json")
    assert sensitivity["primary_pairs_retained_in_main_results"] == 200 and sensitivity["sensitivity_pairs"] == 197
    si = {r["backbone"]: r for r in sensitivity["measurements"] if r["regime"] == "joint" and r["policy"] == "per_image_core13"}
    pair_text += "Omitting three coarse-hash-flagged pairs post hoc gives joint ResNet/CLIP AUC " + f(si["resnet"]["AUC"]) + "/" + f(si["clip"]["AUC"]) + "; all 200 pairs remain primary.\n"
    if any(p(regime, backbone, policy=policy)["FPR"] == 0 for regime in ["joint", "SD_excluded"]
           for backbone in ["resnet", "clip"] for policy in ["per_image_core13", "conformal_all16"]):
        assert all(p("joint", backbone, policy="conformal_all16")["FPR"] == 0 for backbone in ["resnet", "clip"])
        pair_text += r"Each joint probe has S16 identity FPR 0/200; its two-sided 95\% binomial upper bound is 0.0183 under independent common-risk source sampling." + "\n"
    result["PAIRED_CONTENT_RESULTS"] = pair_text

    fair = read("outputs/phase5/fair_watermarks/summary.json")
    assert fair["calibration_photographs"] == fair["test_photographs"] == 400 and fair["test_AI"] == 204 and fair["images"] == 8912
    wi = {(r["method"], r["query_rule"], r["condition"], r["population"]): r for r in fair["measurements"] if r["collection"] == "all" and r["criterion"] == "calibrated_margin"}
    def w(method, population="marked_AI", condition="overall", query="default"):
        return wi[method, query, condition, population]
    result["FAIR_WATERMARK_ROWS"] = "\n".join(" & ".join([label, *[f(w(method, population, condition)["source_mean_rate"])
        for population in ["marked_AI", "photographs"] for method in ["dwt_dct", "trustmark"]]]) + r"\\"
        for condition, label in [("identity", "Identity"), ("jpeg50", "JPEG50"), ("crop", "Crop"), ("lowres32", "Low-res32")])
    result["FAIR_WATERMARK_RESULTS"] = r"""Default positives use the first payload (Table~\ref{tab:modern});
any-three positives average all three payload covers.
Across four views, default-query photographic false-match rates are
""" + f(w("dwt_dct", "photographs")["source_mean_rate"]) + "/" + f(w("trustmark", "photographs")["source_mean_rate"]) + r""" (DWT--DCT/TrustMark), with source-bootstrap intervals
[""" + ", ".join(f(x) for x in w("dwt_dct", "photographs")["source_bootstrap_ci95"]) + "]/[" + ", ".join(f(x) for x in w("trustmark", "photographs")["source_bootstrap_ci95"]) + r"""].
The separately calibrated any-three rule gives marked-AI acceptance
""" + f(w("dwt_dct", query="any3")["source_mean_rate"]) + "/" + f(w("trustmark", query="any3")["source_mean_rate"]) + " and photographic false matches " + f(w("dwt_dct", "photographs", query="any3")["source_mean_rate"]) + "/" + f(w("trustmark", "photographs", query="any3")["source_mean_rate"]) + r""".
Default-query unmarked-AI false matches are
""" + f(w("dwt_dct", "unmarked_AI")["source_mean_rate"]) + "/" + f(w("trustmark", "unmarked_AI")["source_mean_rate"]) + r"""; photographic calibration does not guarantee AI false-match rates.
"""
    zero_collection_rows = [r for r in fair["measurements"] if r["criterion"] == "calibrated_margin"
        and r["population"] == "photographs" and r["collection"] != "all" and r["source_ever_count"] == 0]
    if zero_collection_rows:
        assert all(r["sources"] == 200 for r in zero_collection_rows)
        result["FAIR_WATERMARK_RESULTS"] += r"For any zero-event photographic collection, 0/200 has a two-sided 95\% binomial upper bound 0.0183 under independent common-risk sources; zero bootstrap intervals are degenerate." + "\n"
    native = {(r["method"], r["query_rule"], r["condition"], r["population"]): r for r in fair["measurements"]
              if r["collection"] == "all" and r["criterion"] == "native_rule"}
    low_margin = w("trustmark", condition="lowres32")
    low_native = native["trustmark", "default", "lowres32", "marked_AI"]
    result["FAIR_WATERMARK_RESULTS"] += "TrustMark low-res32 margin acceptance is " + str(low_margin["accepted_images"]) + "/204, but full expected-message/schema/ECC recovery is only " + str(low_native["accepted_images"]) + r"/204; partial information-bit evidence does not imply message recovery." + "\n"
    publication_path = ROOT / "release/publication.json"
    if publication_path.exists():
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
        if publication.get("externally_published") is True:
            url = publication["repository_url"]
            if not url.startswith("https://github.com/") or any(x in url for x in ["{", "}", "\\", " "]):
                raise ValueError("Invalid recorded publication URL")
            result["RELEASE_STATUS"] = "The reviewer code and aggregate evidence are available at " + r"\url{" + url + "}."
        else:
            result["RELEASE_STATUS"] = "The cleaned anonymous code and aggregate evidence are prepared locally; public hosting is pending."
    else:
        result["RELEASE_STATUS"] = "The cleaned anonymous code and aggregate evidence are prepared locally; public hosting is pending."
    return result
