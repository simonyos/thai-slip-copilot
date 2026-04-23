.PHONY: setup synth sample train-detector finetune eval demo serve clean

setup:
	uv venv
	uv pip install -e ".[train,ocr,serve,demo,dev]"

sample:
	uv run python -m thai_slip_copilot.render --out experiments/figures/samples --count 10 --seed 0

synth:
	uv run python -m thai_slip_copilot.render --out data/synth_v1 --count 5000

train-detector:
	uv run python -m thai_slip_copilot.detect_train --name detector_v1

finetune:
	uv run python -m thai_slip_copilot.finetune \
		--base-model google/gemma-3-270m-it \
		--dataset data/synth_v1 \
		--output experiments/runs/gemma3_270m_v1

eval:
	uv run python scripts/eval_on_gold.py \
		--weights experiments/runs/gemma3_270m_v1 \
		--gold data/eval/gold_labels.jsonl

demo:
	uv run streamlit run app/demo.py

serve:
	uv run uvicorn thai_slip_copilot.api:app --reload --port 8000

clean:
	rm -rf data/synth_v* experiments/runs/
