# Lemmazavr

Lemmazavr is a Java-based lemmatizer for Russian words built on top of DeepLearning4J.

The project trains a character-level neural network that maps a word form to its lemma using TSV dictionary data. It includes:

- model training from a morphological dataset
- single-word lemmatization
- configurable training parameters for sequence length, batch size, epochs, and learning rate

The core implementation is located in `src/main/java/org/example` and uses DL4J with the ND4J native backend.

Key classes:

- `LemmatizerModelTrainer` for model training
- `Lemmatizer` for inference with a trained model
- `LemmatizerStrarter` as the application entry point with `main`
