# Lemmazavr

Lemmazavr is a Java-based lemmatizer for Russian words built on top of DeepLearning4J.

The project trains a character-level neural network that maps a word form to its lemma using TSV dictionary data. It includes:

- model training from a morphological dataset
- single-word lemmatization
- batch processing from text files
- configurable training parameters for sequence length, batch size, epochs, and learning rate

The core implementation is located in `src/main/java/org/example` and uses DL4J with the ND4J native backend.
