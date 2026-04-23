package org.example;

/**
 * Конфигурация для обучения и инференса модели лемматизации.
 */
public class LemmatizerConfig {

    /** Максимальная длина слова (в символах) */
    private final int maxLen;

    /** Размер батча для обучения (уменьшен для экономии памяти) */
    private final int batchSize;

    /** Количество эпох обучения */
    private final int epochs;

    /** Размер LSTM слоя (уменьшен для экономии памяти) */
    private final int lstmSize;

    /** Скорость обучения */
    private final double learningRate;

    /** Seed для воспроизводимости результатов */
    private final long seed;

    /** Путь к файлу датасета (TSV формат: словоформа\tлемма) */
    private final String datasetPath;

    /** Путь для сохранения обученной модели */
    private final String modelOutputPath;

    private LemmatizerConfig(Builder builder) {
        this.maxLen = builder.maxLen;
        this.batchSize = builder.batchSize;
        this.epochs = builder.epochs;
        this.lstmSize = builder.lstmSize;
        this.learningRate = builder.learningRate;
        this.seed = builder.seed;
        this.datasetPath = builder.datasetPath;
        this.modelOutputPath = builder.modelOutputPath;
    }

    public static Builder builder() {
        return new Builder();
    }

    public int getMaxLen() {
        return maxLen;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getEpochs() {
        return epochs;
    }

    public int getLstmSize() {
        return lstmSize;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public long getSeed() {
        return seed;
    }

    public String getDatasetPath() {
        return datasetPath;
    }

    public String getModelOutputPath() {
        return modelOutputPath;
    }

    /**
     * Валидация конфигурации.
     * @throws IllegalArgumentException если параметры некорректны
     */
    public void validate() {
        if (maxLen <= 0) throw new IllegalArgumentException("maxLen должен быть > 0");
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize должен быть > 0");
        if (epochs <= 0) throw new IllegalArgumentException("epochs должен быть > 0");
        if (lstmSize <= 0) throw new IllegalArgumentException("lstmSize должен быть > 0");
        if (learningRate <= 0 || learningRate > 1) throw new IllegalArgumentException("learningRate должен быть в диапазоне (0, 1]");
        if (datasetPath == null || datasetPath.trim().isEmpty()) throw new IllegalArgumentException("datasetPath не может быть пустым");
        if (modelOutputPath == null || modelOutputPath.trim().isEmpty()) throw new IllegalArgumentException("modelOutputPath не может быть пустым");
    }

    public static class Builder {
        private int maxLen = 30;
        private int batchSize = 32;
        private int epochs = 10;
        private int lstmSize = 64;
        private double learningRate = 0.002;
        private long seed = 42L;
        private String datasetPath;
        private String modelOutputPath;

        public Builder maxLen(int maxLen) {
            this.maxLen = maxLen;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder lstmSize(int lstmSize) {
            this.lstmSize = lstmSize;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder datasetPath(String datasetPath) {
            this.datasetPath = datasetPath;
            return this;
        }

        public Builder modelOutputPath(String modelOutputPath) {
            this.modelOutputPath = modelOutputPath;
            return this;
        }

        public LemmatizerConfig build() {
            return new LemmatizerConfig(this);
        }
    }
}
