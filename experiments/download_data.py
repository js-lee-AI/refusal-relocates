import argparse
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download the public source datasets")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    urls = {
        "alpaca.json": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "advbench.csv": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
        "xstest.csv": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    }
    for name, url in urls.items():
        target = output / name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
    target = output / "pku.jsonl"
    if not target.exists():
        from datasets import load_dataset
        load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train").to_json(str(target), force_ascii=False)
    print(f"Downloaded source data to {output}")


if __name__ == "__main__":
    main()
