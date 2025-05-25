#----------------------------------------
# Targets to run the model pipeline
#----------------------------------------

# Download the data
download:
	python -m src.data.download

# Preprocess the data
preprocess:
	python -m src.preprocess.dataset

# Train the model
train:
	python -m src.model.train

# Evaluate the model
evaluate:
	python -m src.evaluate.evaluate

# Visualize the model
visualize:
	python -m src.visualization.visualize

# Run the model pipeline
run: preprocess train evaluate visualize

# Run all: Run the model pipeline
all: download preprocess train evaluate visualize

#----------------------------------------
# Cleaning folders
#----------------------------------------

# Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Delete all data
clean-data:
	rm -rf data/names/ 
	rm -rf data/data.zip
	rm -rf data/eng-fra.txt
	rm -rf models/model.pth
	rm -rf reports/figures/confusion.png

# Delete all
clean-all: clean clean-data

# NOTE FOR WINDOWS USERS: Make is more challenging to install on Windows.
# I recommend using Chocolately guide here: 
# https://earthly.dev/blog/makefiles-on-windows/
