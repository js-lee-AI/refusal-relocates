"""Section 6.2 and Appendix I. The flatness detector against top-two repair failures.

The score of an adapter is the mean participation ratio of its attention-projection updates over
the LoRA rank. The threshold is calibrated on 75 benign Llama-3.1-8B fine-tunes to a 5% false-
positive budget, then applied to 27 dose-100 attacks on the same 50 AdvBench prompts.
"""

from refusal.spectrum import calibrate_detector

from _adaptive import CLEAN, adapters, detector_set, landed, projected, repair_failed, spread
from _records import load, number, parser, show


def summary(records=None):
    benign = [r["detector_score"] for r in load("benign_spectra.csv", records)]
    attacks = detector_set(adapters(records))
    result = calibrate_detector(benign, [r["detector_score"] for r in attacks], budget=0.05)
    threshold = result["threshold"]
    flag = {id(r): f for r, f in zip(attacks, result["attack_flags"])}
    hits = [r for r in attacks if landed(r)]
    failures = [r for r in hits if repair_failed(r)]
    flagged = [r for r in failures if flag[id(r)]]
    missed = [r for r in failures if not flag[id(r)]]
    over = [r for r in failures if r["common_repaired_over_refusal"] - r["common_clean_over_refusal"] > 0.10]
    return {
        "benign adapters": len(benign),
        "benign maximum": number(max(benign), 3, lead=True),
        "threshold at a 5% budget": number(threshold, 3, lead=True),
        "benign false positives": f"{round(result['benign_fpr'] * len(benign))}/{len(benign)}",
        "attacks": len(attacks),
        "attacks that land": len(hits),
        "top-two repair failures": len(failures),
        "failures with over-refusal rise above 0.10": len(over),
        "failures flagged": f"{len(flagged)} ({', '.join(sorted({r['mode'] for r in flagged}))})",
        "failures missed": f"{sum(r['mode'] == 'standard' for r in missed)} ordinary, "
                           f"{sum(r['mode'] == 'projected' for r in missed)} projected",
        "missed scores": f"{number(min(r['detector_score'] for r in missed), 3, lead=True)} to "
                         f"{number(max(r['detector_score'] for r in missed), 3, lead=True)}",
        "spread attack scores": ", ".join(number(r["detector_score"], 3, lead=True) for r in spread(adapters(records))),
        "repair threshold, 80% of clean": number(0.8 * CLEAN, 3, lead=True),
        "projected attacks that land": sum(landed(r) for r in projected(adapters(records))),
    }


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    show("Section 6.2 and Appendix I. The flatness detector.", ["quantity", "value"],
         [[key, value] for key, value in summary(args.records).items()], args.out)
