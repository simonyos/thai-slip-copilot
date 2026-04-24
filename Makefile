.PHONY: setup synth sample train-detector gold verify finetune eval demo serve clean

setup:
	uv venv
	uv pip install -e ".[train,ocr,serve,demo,dev]"

sample:
	uv run python -m thai_slip_copilot.render --out experiments/figures/samples --count 10 --seed 0

synth:
	uv run python -m thai_slip_copilot.render --out data/synth_v1 --count 5000

train-detector:
	uv run python -m thai_slip_copilot.detect_train --name detector_v1

gold:
	uv run python scripts/build_gold_set.py \
		--weights experiments/runs/detector_v2_1/weights/best.pt \
		--images /Volumes/SM-EXT/Documents/Slip-orc-sample \
		--out data/eval/gold_phone_v1.jsonl --n 50

verify:
	uv run streamlit run app/verify_slip_labels.py

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
