package org.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class LemmatizerDataSetIterator implements DataSetIterator {

    private static final char END_CHAR = '\1';

    private final Path datasetPath;
    private final int batchSize;
    private final int maxLen;
    private final Map<Character, Integer> charToIndex;
    private final int vocabSize;

    private transient BufferedReader reader;
    private transient DataSet nextDataSet;
    private transient boolean finished;

    private DataSetPreProcessor preProcessor;

    public LemmatizerDataSetIterator(
            Path datasetPath,
            int batchSize,
            int maxLen,
            Map<Character, Integer> charToIndex,
            int vocabSize
    ) {
        this.datasetPath = Objects.requireNonNull(datasetPath);
        this.batchSize = batchSize;
        this.maxLen = maxLen;
        this.charToIndex = Objects.requireNonNull(charToIndex);
        this.vocabSize = vocabSize;
        reset();
    }

    @Override
    public boolean hasNext() {
        if (nextDataSet == null && !finished) {
            nextDataSet = loadNextBatch();
        }
        return nextDataSet != null;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public DataSet next(int num) {
        if (nextDataSet == null && !finished) {
            nextDataSet = loadNextBatch(num);
        }

        if (nextDataSet == null) {
            throw new NoSuchElementException("Больше нет данных");
        }

        DataSet current = nextDataSet;
        nextDataSet = null;

        if (preProcessor != null) {
            preProcessor.preProcess(current);
        }

        return current;
    }

    private DataSet loadNextBatch() {
        return loadNextBatch(batchSize);
    }

    private DataSet loadNextBatch(int requestedBatchSize) {
        List<String[]> rows = new ArrayList<>(requestedBatchSize);

        try {
            String line;
            while (rows.size() < requestedBatchSize && (line = reader.readLine()) != null) {
                String[] parts = splitTsvLine(line);
                if (parts != null) {
                    rows.add(parts);
                }
            }

            if (rows.isEmpty()) {
                finished = true;
                closeReader();
                return null;
            }

            return toDataSet(rows);

        } catch (IOException e) {
            throw new UncheckedIOException("Ошибка чтения датасета: " + datasetPath, e);
        }
    }

    private String[] splitTsvLine(String line) {
        if (line == null || line.isBlank()) {
            return null;
        }

        int tabIndex = line.indexOf('\t');
        if (tabIndex <= 0 || tabIndex >= line.length() - 1) {
            return null;
        }

        String word = line.substring(0, tabIndex).trim();
        String lemmaAndTags = line.substring(tabIndex + 1).trim();
        int secondTabIndex = lemmaAndTags.indexOf('\t');
        String lemma = secondTabIndex >= 0
                ? lemmaAndTags.substring(0, secondTabIndex).trim()
                : lemmaAndTags;

        if (word.isEmpty() || lemma.isEmpty()) {
            return null;
        }

        return new String[]{word, lemma};
    }

    private DataSet toDataSet(List<String[]> rows) {
        int currentBatchSize = rows.size();

        INDArray features = Nd4j.zeros(DataType.FLOAT, currentBatchSize, vocabSize, maxLen);
        INDArray labels = Nd4j.zeros(DataType.FLOAT, currentBatchSize, vocabSize, maxLen);

        INDArray labelsMask = Nd4j.zeros(DataType.FLOAT, currentBatchSize, maxLen);

        for (int i = 0; i < currentBatchSize; i++) {
            String word = rows.get(i)[0];
            String lemma = rows.get(i)[1];

            fillInput(features, i, word);
            fillLabel(labels, labelsMask, i, lemma);
        }

        return new DataSet(features, labels, null, labelsMask);
    }

    private void fillInput(INDArray arr, int batchIdx, String text) {
        int len = Math.min(text.length(), maxLen);

        for (int t = 0; t < len; t++) {
            int charIdx = charToIndex.getOrDefault(text.charAt(t), 0);
            arr.putScalar(new int[]{batchIdx, charIdx, t}, 1.0f);
        }

        // padding
        for (int t = len; t < maxLen; t++) {
            arr.putScalar(new int[]{batchIdx, 0, t}, 1.0f);
        }
    }

    private void fillLabel(INDArray arr, INDArray mask, int batchIdx, String text) {
        int len = Math.min(text.length(), Math.max(0, maxLen - 1));

        for (int t = 0; t < len; t++) {
            int charIdx = charToIndex.getOrDefault(text.charAt(t), 0);
            arr.putScalar(new int[]{batchIdx, charIdx, t}, 1.0f);
            mask.putScalar(new int[]{batchIdx, t}, 1.0f);
        }

        if (len < maxLen) {
            arr.putScalar(new int[]{batchIdx, charToIndex.get(END_CHAR), len}, 1.0f);
            mask.putScalar(new int[]{batchIdx, len}, 1.0f);
        }
    }

    // ===== DL4J API =====

    @Override
    public int inputColumns() {
        return vocabSize;
    }

    @Override
    public int totalOutcomes() {
        return vocabSize;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        closeReader();
        try {
            this.reader = Files.newBufferedReader(datasetPath, StandardCharsets.UTF_8);
            this.finished = false;
            this.nextDataSet = null;
        } catch (IOException e) {
            throw new UncheckedIOException("Не удалось открыть файл: " + datasetPath, e);
        }
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    // ===== util =====

    private void closeReader() {
        if (reader != null) {
            try {
                reader.close();
            } catch (IOException ignored) {
            }
            reader = null;
        }
    }
}
