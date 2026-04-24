package org.example;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Общая логика построения и восстановления словаря символов.
 */
@Getter
@Slf4j
@AllArgsConstructor(access = AccessLevel.PRIVATE)
final class LemmatizerVocabulary {

    static final char PADDING_CHAR = '\0';
    static final char END_CHAR = '\1';
    static final String VOCABULARY_METADATA_KEY = "lemmatizerVocabulary";

    private final Map<Character, Integer> charToIndex;
    private final Map<Integer, Character> indexToChar;
    private final int vocabSize;
    private final long sampleCount;

    static LemmatizerVocabulary buildFromDataset(Path datasetPath) throws IOException {
        Set<Character> chars = new TreeSet<>();
        chars.add(PADDING_CHAR);
        chars.add(END_CHAR);

        long validRows = 0;
        try (var lines = Files.lines(datasetPath, StandardCharsets.UTF_8)) {
            Iterator<String> iterator = lines.iterator();
            while (iterator.hasNext()) {
                WordPair pair = parseWordPair(iterator.next());
                if (pair == null) {
                    continue;
                }
                validRows++;
                pair.wordForm().chars().mapToObj(c -> (char) c).forEach(chars::add);
                pair.lemma().chars().mapToObj(c -> (char) c).forEach(chars::add);
            }
        }

        return fromChars(chars, validRows);
    }

    static LemmatizerVocabulary rebuildFromModel(File modelFile) throws IOException {
        VocabularyMetadata metadata = tryReadMetadata(modelFile);
        if (metadata != null && metadata.getCharToIndex() != null && !metadata.getCharToIndex().isEmpty()) {
            Map<Character, Integer> charToIndex = new HashMap<>(metadata.getCharToIndex());
            Map<Integer, Character> indexToChar = rebuildIndexToChar(charToIndex);
            int vocabSize = metadata.getVocabSize() > 0 ? metadata.getVocabSize() : charToIndex.size();

            log.info("Словарь восстановлен из model metadata. Размер: {}", vocabSize);
            return new LemmatizerVocabulary(charToIndex, indexToChar, vocabSize, 0);
        }

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile, false);

        int vocabSize;
        try {
            org.deeplearning4j.nn.conf.layers.Layer layer0 = model.getLayerWiseConfigurations().getConf(0).getLayer();
            if (layer0 instanceof FeedForwardLayer feedForwardLayer) {
                vocabSize = Math.toIntExact(feedForwardLayer.getNIn());
            } else {
                throw new IllegalStateException("Первый слой модели не является FeedForwardLayer: " + layer0.getClass().getName());
            }
        } catch (Exception e) {
            throw new IllegalStateException("Не удалось восстановить размер словаря из модели: " + modelFile.getAbsolutePath(), e);
        }

        Map<Character, Integer> charToIndex = new HashMap<>();
        Map<Integer, Character> indexToChar = new HashMap<>();
        for (int i = 0; i < vocabSize; i++) {
            char c = (char) (' ' + i);
            charToIndex.put(c, i);
            indexToChar.put(i, c);
        }

        log.warn("Словарь восстановлен только по размеру nIn = {}. Для корректной лемматизации переобучите модель и сохраните её с metadata словаря.", vocabSize);
        return new LemmatizerVocabulary(charToIndex, indexToChar, vocabSize, 0);
    }

    VocabularyMetadata toMetadata() {
        return new VocabularyMetadata(new HashMap<>(charToIndex), vocabSize);
    }

    private static LemmatizerVocabulary fromChars(Set<Character> chars, long sampleCount) {
        Map<Character, Integer> charToIndex = new HashMap<>();
        Map<Integer, Character> indexToChar = new HashMap<>();

        int idx = 0;
        for (char c : chars) {
            charToIndex.put(c, idx);
            indexToChar.put(idx, c);
            idx++;
        }

        return new LemmatizerVocabulary(charToIndex, indexToChar, chars.size(), sampleCount);
    }

    private static Map<Integer, Character> rebuildIndexToChar(Map<Character, Integer> charToIndex) {
        Map<Integer, Character> indexToChar = new HashMap<>();
        for (Map.Entry<Character, Integer> entry : charToIndex.entrySet()) {
            indexToChar.put(entry.getValue(), entry.getKey());
        }
        return indexToChar;
    }

    private static VocabularyMetadata tryReadMetadata(File modelFile) {
        try {
            return ModelSerializer.getObjectFromFile(modelFile, VOCABULARY_METADATA_KEY);
        } catch (Exception e) {
            log.warn("Metadata словаря не найдена в модели {}. Используем fallback по nIn.", modelFile.getAbsolutePath());
            return null;
        }
    }

    private static WordPair parseWordPair(String line) {
        if (line == null) {
            return null;
        }

        String trimmed = line.trim();
        if (trimmed.isEmpty() || trimmed.startsWith("#")) {
            return null;
        }

        String[] parts = trimmed.split("\t");
        if (parts.length < 2) {
            return null;
        }

        String wordForm = parts[0].trim();
        String lemma = parts[1].trim();
        if (wordForm.isEmpty() || lemma.isEmpty()) {
            return null;
        }

        return new WordPair(wordForm, lemma);
    }

    private record WordPair(String wordForm, String lemma) {
    }

    @Getter
    @AllArgsConstructor
    static class VocabularyMetadata implements Serializable {
        private static final long serialVersionUID = 1L;

        private final Map<Character, Integer> charToIndex;
        private final int vocabSize;
    }
}
