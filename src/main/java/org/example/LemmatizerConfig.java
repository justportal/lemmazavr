package org.example;

import lombok.Builder;
import lombok.Getter;

/**
 * Конфигурация для обучения и инференса модели лемматизации.
 */
@Getter
@Builder
public class LemmatizerConfig {

    /** Максимальная длина слова (в символах) */
    @Builder.Default
    private final int maxLen = 30;

    /** Размер батча для обучения (уменьшен для экономии памяти) */
    @Builder.Default
    private final int batchSize = 32;

    /** Количество эпох обучения */
    @Builder.Default
    private final int epochs = 10;

    /** Размер LSTM слоя (уменьшен для экономии памяти) */
    @Builder.Default
    private final int lstmSize = 64;

    /** Скорость обучения */
    @Builder.Default
    private final double learningRate = 0.002;

    /** Порог gradient clipping для стабилизации обучения */
    @Builder.Default
    private final double gradientClipThreshold = 1.0;

    /** Seed для воспроизводимости результатов */
    @Builder.Default
    private final long seed = 42L;

    /** Путь к файлу датасета (TSV формат: словоформа\tлемма) */
    private final String datasetPath;

    /** Путь для сохранения обученной модели */
    private final String modelOutputPath;

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
        if (gradientClipThreshold <= 0) throw new IllegalArgumentException("gradientClipThreshold должен быть > 0");
        if (datasetPath == null || datasetPath.trim().isEmpty()) throw new IllegalArgumentException("datasetPath не может быть пустым");
        if (modelOutputPath == null || modelOutputPath.trim().isEmpty()) throw new IllegalArgumentException("modelOutputPath не может быть пустым");
    }
}
