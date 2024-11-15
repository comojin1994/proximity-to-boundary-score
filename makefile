setup:
	python datasets/setups/bcic2a.py

train:
	nohup python training.py > logs/train_output.log 2> logs/train_error.log &

evaluate:
	nohup papermill evaluation.ipynb logs/evaluation_output.ipynb > logs/evaluate_output.log 2> logs/evaluate_error.log &