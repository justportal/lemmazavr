package org.example;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

/**
 * Класс для использования обученной модели лемматизации.
 */
@Slf4j
public class Lemmatizer {

    private static final int MAX_LEN_DEFAULT = 30;

    private final LemmatizerConfig config;
    private LemmatizerVocabulary vocabulary;

    public Lemmatizer() {
        this(null);
    }

    public Lemmatizer(LemmatizerConfig config) {
        this.config = config;
    }

    /**
     * Выполняет лемматизацию слова.
     *
     * @param word входное слово
     * @param modelPath путь к файлу модели
     * @return лемма или исходное слово при ошибке
     */
    public String lemmatize(String word, String modelPath) {
        if (word == null || word.trim().isEmpty()) {
            log.warn("Получена пустая строка. Возврат исходного значения.");
            return word;
        }

        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            throw new IllegalArgumentException("Файл модели не найден: " + modelPath);
        }

        try {
            if (vocabulary == null) {
                vocabulary = LemmatizerVocabulary.rebuildFromModel(modelFile);
            }

            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile, true);
            INDArray input = encodeInput(word);
            INDArray output = model.output(input);
            String predictedLemma = decodeOutput(output);

            if (predictedLemma.isEmpty()) {
                log.warn("Модель не смогла предсказать лемму для '{}'. Возврат исходного слова.", word);
                return word;
            }

            return predictedLemma;
        } catch (IOException e) {
            log.error("Ошибка при загрузке/инференсе модели: {}", e.getMessage(), e);
            throw new RuntimeException("Не удалось выполнить лемматизацию", e);
        }
    }

    private INDArray encodeInput(String word) {
        int maxLen = config != null ? config.getMaxLen() : MAX_LEN_DEFAULT;
        INDArray input = Nd4j.zeros(1, vocabulary.getVocabSize(), maxLen);
        int len = Math.min(word.length(), maxLen);

        for (int t = 0; t < maxLen; t++) {
            int charIdx = (t < len) ? vocabulary.getCharToIndex().getOrDefault(word.charAt(t), 0) : 0;
            input.putScalar(new int[]{0, charIdx, t}, 1.0);
        }

        return input;
    }

    private String decodeOutput(INDArray output) {
        INDArray predictedIndices = Nd4j.argMax(output, 1);
        double[] indices = predictedIndices.data().asDouble();

        StringBuilder sb = new StringBuilder();
        for (double idx : indices) {
            int charIdx = (int) idx;
            if (charIdx == 0) {
                break;
            }

            char predictedChar = vocabulary.getIndexToChar().getOrDefault(charIdx, '?');
            if (predictedChar == LemmatizerVocabulary.END_CHAR) {
                break;
            }

            sb.append(predictedChar);
        }

        return sb.toString();
    }
}
