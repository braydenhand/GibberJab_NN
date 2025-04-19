# GibberJab_NN
This is an implementation of the GibberJab neural network latent embedding language. 


To use: 
create a conda env, activate, and install requirements 

conda create --name GibberJab_NN python=3.10 pip
conda activate GibberJab_NN
pip install -r requirements.txt

python neural_codec.py                       # Train from scratch if no model exists
python neural_codec.py --retrain             # Force retraining
python neural_codec.py --epochs 50           # Set training epochs
python neural_codec.py --corpus_size 10000000 # Use larger corpus
python neural_codec.py --test_only           # Skip training and just test