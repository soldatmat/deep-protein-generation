import tensorflow as tf

import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint

from models.vaes import MSAVAE
from utils.io import load_gzdata
from utils.data_loaders import one_hot_generator

# Define training parameters
batch_size = 32
seed = 0
n_epochs = 14
verbose = 1
save_all_epochs = False

data_name = "msa_Plantae_sesq-14.11.2024-noX.fasta.gz"
# data_name = "cluster_79.aln-fasta.gz"

seed = np.random.seed(seed)

# Load aligned sequences
_, msa_seqs = load_gzdata("data/training_data/" + data_name, one_hot=False)
_, val_msa_seqs = load_gzdata("data/training_data/" + data_name, one_hot=False)
# _, msa_seqs = load_gzdata('data/training_data/luxafilt_llmsa_train.fa.gz', one_hot=False)
# _, val_msa_seqs = load_gzdata('data/training_data/luxafilt_llmsa_val.fa.gz', one_hot=False)

# Preprocess sequences
# msa_seqs = [s[0:360] for s in msa_seqs]
# val_msa_seqs = [s[0:360] for s in val_msa_seqs]

""" def convert_seqs(seqs):
    return [s[0:360] for s in seqs]
msa_seqs = convert_seqs(msa_seqs)
val_msa_seqs = convert_seqs(val_msa_seqs) """

print("Loaded %d training sequences with length %d." % (len(msa_seqs), len(msa_seqs[0])))

# Define data generators
train_gen = one_hot_generator(msa_seqs, padding=None)
val_gen = one_hot_generator(val_msa_seqs, padding=None)

# Define model
print('Building model')
# input_dim = 360
input_dim = len(msa_seqs[0])
print("Using input dimension", input_dim)
model = MSAVAE(original_dim=input_dim, latent_dim=10)

# (Optionally) define callbacks
callbacks = [CSVLogger("output/logs/msavae-" + data_name + ".aln-fasta.csv")]

if save_all_epochs:
    callbacks.append(
        ModelCheckpoint(
            "output/weights/" + data_name + "/msavae"
            + ".{epoch:02d}-{luxa_errors_mean:.2f}.hdf5",
            save_best_only=False,
            verbose=1,
        )
    )

print('Training model')
# Train model https://github.com/keras-team/keras/issues/8595
model.VAE.fit_generator(generator=train_gen,
                        steps_per_epoch=len(msa_seqs) // batch_size,
                        verbose=verbose,
                        epochs=n_epochs,
                        validation_data=val_gen,
                        validation_steps=len(val_msa_seqs) // batch_size,
                        callbacks=callbacks)

if not save_all_epochs:
    model.save_weights("output/weights/msavae-" + data_name + ".aln-fasta.h5")
