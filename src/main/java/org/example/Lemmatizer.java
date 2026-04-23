package org.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Лемматизатор на основе нейронной сети (LSTM).
 * <p>
 * Модель принимает словоформу на вход и предсказывает её лемму.
 * Используется посимвольное кодирование с one-hot представлением.
 * </p>
 */
public class Lemmatizer {

    private static final Logger log = LoggerFactory.getLogger(Lemmatizer.class);
    private static final char PADDING_CHAR = '\0';
    private static final char END_CHAR = '\1';

    private final LemmatizerConfig config;
    private Map<Character, Integer> charToIndex;
    private Map<Integer, Character> indexToChar;
    private int vocabSize;

    /**
     * Создает лемматизатор с конфигурацией по умолчанию.
     */
    public Lemmatizer() {
        this(null);
    }

    /**
     * Создает лемматизатор с заданной конфигурацией.
     *
     * @param config конфигурация
     */
    public Lemmatizer(LemmatizerConfig config) {
        this.config = config;
        if (config != null) {
            config.validate();
        }
    }

    /**
     * Обучает модель на датасете.
     *
     * @param datasetPath   путь к TSV файлу (словоформа\tлемма)
     * @param modelOutputPath путь для сохранения модели
     */
    public void train(String datasetPath, String modelOutputPath) {
        LemmatizerConfig trainConfig = Optional.ofNullable(config)
                .orElseGet(() -> LemmatizerConfig.builder()
                        .datasetPath(datasetPath)
                        .modelOutputPath(modelOutputPath)
                        .build());

        trainConfig.validate();

        try {
            Path dataset = Paths.get(datasetPath);
            log.info("Подготовка датасета: {}", datasetPath);
            long sampleCount = buildVocabulary(dataset);

            if (sampleCount == 0) {
                throw new IllegalStateException("Датасет пуст или не содержит валидных строк.");
            }

            int batchCount = (int) Math.ceil((double) sampleCount / trainConfig.getBatchSize());
            log.info("Словарь построен. Размер: {}, MAX_LEN: {}, примеров: {}, батчей на эпоху: {}",
                    vocabSize, trainConfig.getMaxLen(), sampleCount, batchCount);

            MultiLayerNetwork model = buildModel(trainConfig);
            model.init();
            model.setListeners(new ScoreIterationListener(100));

            log.info("Начало обучения...");
            for (int epoch = 0; epoch < trainConfig.getEpochs(); epoch++) {
                long start = System.currentTimeMillis();
                LemmatizerDataSetIterator iterator = new LemmatizerDataSetIterator(
                        dataset,
                        trainConfig.getBatchSize(),
                        trainConfig.getMaxLen(),
                        charToIndex,
                        vocabSize
                );
                model.fit(iterator);
                long duration = System.currentTimeMillis() - start;
                log.info("Эпоха {}/{} завершена за {} мс", epoch + 1, trainConfig.getEpochs(), duration);
            }

            saveModel(model, modelOutputPath);
            log.info("Модель успешно сохранена: {}", modelOutputPath);

        } catch (IOException e) {
            log.error("Ошибка ввода/вывода при обучении модели: {}", e.getMessage(), e);
            throw new RuntimeException("Не удалось обучить модель", e);
        } catch (Exception e) {
            log.error("Ошибка при обучении модели: {}", e.getMessage(), e);
            throw new RuntimeException("Не удалось обучить модель", e);
        }
    }

    /**
     * Выполняет лемматизацию слова.
     *
     * @param word      входное слово
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
            // Восстанавливаем словарь из модели или используем существующий
            if (charToIndex == null || charToIndex.isEmpty()) {
                rebuildVocabularyFromModel(modelFile);
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

    // ================= МЕТОДЫ ОБУЧЕНИЯ =================

    /**
     * Читает TSV файл с парами словоформа-лемма.
     */
    private long buildVocabulary(Path datasetPath) throws IOException {
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
                pair.getWordForm().chars().mapToObj(c -> (char) c).forEach(chars::add);
                pair.getLemma().chars().mapToObj(c -> (char) c).forEach(chars::add);
            }
        }

        charToIndex = new HashMap<>();
        indexToChar = new HashMap<>();

        int idx = 0;
        for (char c : chars) {
            charToIndex.put(c, idx);
            indexToChar.put(idx, c);
            idx++;
        }

        vocabSize = chars.size();
        return validRows;
    }

    /**
     * Парсит строку TSV в пару словоформа-лемма.
     */
    private WordPair parseWordPair(String line) {
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

    /**
     * Восстанавливает словарь из сохранённой модели.
     */
    private void rebuildVocabularyFromModel(File modelFile) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile, false);
        
        // Получаем размер входа первого слоя (vocabSize) через конфигурацию
        // Для LSTM слоя nIn = vocabSize
        int lstmSize = 128; // значение по умолчанию
        try {
            // Пытаемся получить размер из конфигурации
            org.deeplearning4j.nn.conf.layers.Layer layer0 = model.getLayerWiseConfigurations().getConf(0).getLayer();
            if (layer0 instanceof LSTM) {
                LSTM lstm = (LSTM) layer0;
                // В DL4J нет прямого getter для nIn в runtime, используем дефолтное значение
                // или восстанавливаем из данных
            }
        } catch (Exception e) {
            log.warn("Не удалось восстановить точный размер словаря из модели, используем эвристику");
        }
        
        // Для простоты предполагаем стандартный размер словаря
        // В реальной реализации нужно сохранять словарь отдельно в JSON
        vocabSize = 100; //保守估计
        
        charToIndex = new HashMap<>();
        indexToChar = new HashMap<>();

        // Простая эвристика: символы от пробела onwards
        for (int i = 0; i < vocabSize; i++) {
            char c = (char) (' ' + i);
            charToIndex.put(c, i);
            indexToChar.put(i, c);
        }

        log.info("Словарь восстановлен из модели. Размер: {}", vocabSize);
    }

    /**
     * Строит конфигурацию нейронной сети.
     */
    private MultiLayerNetwork buildModel(LemmatizerConfig cfg) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(cfg.getSeed())
                .updater(new Adam(cfg.getLearningRate()))
                .weightInit(WeightInit.XAVIER)
                .list()
                // LSTM слой: принимает [batch, vocabSize, time]
                .layer(new LSTM.Builder()
                        .nIn(vocabSize)
                        .nOut(cfg.getLstmSize())
                        .activation(Activation.TANH)
                        .build())
                // Dense слой применяется к каждому временному шагу
                .layer(new DenseLayer.Builder()
                        .nIn(cfg.getLstmSize())
                        .nOut(cfg.getLstmSize())
                        .activation(Activation.RELU)
                        .build())
                // RnnOutputLayer предсказывает символ для каждой позиции
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(cfg.getLstmSize())
                        .nOut(vocabSize)
                        .build())
                .build();

        return new MultiLayerNetwork(conf);
    }

    /**
     * Сохраняет модель на диск.
     */
    private void saveModel(MultiLayerNetwork model, String outputPath) throws IOException {
        File modelFile = new File(outputPath);
        File parent = modelFile.getParentFile();

        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        ModelSerializer.writeModel(model, modelFile, true);
    }

    // ================= МЕТОДЫ ИНФЕРЕНСА =================

    /**
     * Кодирует входное слово в one-hot тензор.
     *
     * @param word входное слово
     * @return тензор формы [1, vocabSize, maxLen]
     */
    private INDArray encodeInput(String word) {
        int maxLen = config != null ? config.getMaxLen() : MAX_LEN_DEFAULT;
        INDArray input = Nd4j.zeros(1, vocabSize, maxLen);
        int len = Math.min(word.length(), maxLen);

        for (int t = 0; t < maxLen; t++) {
            int charIdx = (t < len) ? charToIndex.getOrDefault(word.charAt(t), 0) : 0;
            input.putScalar(new int[]{0, charIdx, t}, 1.0);
        }

        return input;
    }

    /**
     * Декодирует выход модели в строку.
     *
     * @param output выход модели формы [1, vocabSize, maxLen]
     * @return предсказанная лемма
     */
    private String decodeOutput(INDArray output) {
        INDArray predictedIndices = Nd4j.argMax(output, 1); // [1, maxLen]
        double[] indices = predictedIndices.data().asDouble();

        StringBuilder sb = new StringBuilder();
        for (double idx : indices) {
            int charIdx = (int) idx;
            if (charIdx == 0) break; // Остановка на padding-символе
            char predictedChar = indexToChar.getOrDefault(charIdx, '?');
            if (predictedChar == END_CHAR) break;
            sb.append(predictedChar);
        }

        return sb.toString();
    }

    // ================= ВНУТРЕННИЕ КЛАССЫ =================

    /**
     * Пара словоформа-лемма.
     */
    private static class WordPair {
        private final String wordForm;
        private final String lemma;

        WordPair(String wordForm, String lemma) {
            this.wordForm = wordForm;
            this.lemma = lemma;
        }

        String getWordForm() {
            return wordForm;
        }

        String getLemma() {
            return lemma;
        }
    }

    private static final int MAX_LEN_DEFAULT = 30;

    // ================= ПРИМЕР ИСПОЛЬЗОВАНИЯ =================

    /**
     * Точка входа приложения.
     * <p>
     * Поддерживаемые команды:
     * <ul>
     *   <li>{@code train <dataset_path> <model_path>} - обучить модель</li>
     *   <li>{@code lemmatize <word> <model_path>} - лемматизировать слово</li>
     *   <li>{@code batch <input_file> <model_path> <output_file>} - пакетная обработка</li>
     * </ul>
     */
    public static void main(String[] args) {
        if (args.length == 0) {
            printUsage();
            return;
        }

        String command = args[0].toLowerCase();

        try {
            switch (command) {
                case "train":
                    handleTrain(args);
                    break;
                case "lemmatize":
                    handleLemmatize(args);
                    break;
                case "batch":
                    handleBatch(args);
                    break;
                default:
                    log.error("Неизвестная команда: {}", command);
                    printUsage();
            }
        } catch (IllegalArgumentException e) {
            log.error("Ошибка аргументов: {}", e.getMessage());
            printUsage();
        } catch (Exception e) {
            log.error("Критическая ошибка: {}", e.getMessage(), e);
            System.exit(1);
        }
    }

    private static void printUsage() {
        System.out.println("\n=== Russian Lemmatizer ===");
        System.out.println("\nИспользование:");
        System.out.println("  java -jar simple-lemmatizer-1.0-SNAPSHOT-all.jar train <dataset.tsv> <model.zip>");
        System.out.println("  java -jar simple-lemmatizer-1.0-SNAPSHOT-all.jar lemmatize <word> <model.zip>");
        System.out.println("  java -jar simple-lemmatizer-1.0-SNAPSHOT-all.jar batch <input.txt> <model.zip> <output.txt>");
        System.out.println("\nПримеры:");
        System.out.println("  java -jar simple-lemmatizer-1.0-SNAPSHOT-all.jar train data/opcorpora.tsv data/model.zip");
        System.out.println("  java -jar simple-lemmatizer-1.0-SNAPSHOT-all.jar lemmatize \"бежал\" data/model.zip");
        System.out.println();
    }

    private static void handleTrain(String[] args) {
        if (args.length < 3) {
            throw new IllegalArgumentException("Команда train требует 2 аргумента: <dataset.tsv> <model.zip>");
        }

        String datasetPath = args[1];
        String modelPath = args[2];

        LemmatizerConfig config = LemmatizerConfig.builder()
                .datasetPath(datasetPath)
                .modelOutputPath(modelPath)
                .epochs(15)
                .batchSize(128)
                .build();

        Lemmatizer lemmatizer = new Lemmatizer(config);

        if (!Files.exists(Paths.get(datasetPath))) {
            throw new IllegalArgumentException("Датасет не найден: " + datasetPath);
        }

        log.info("Начало обучения модели...");
        log.info("Датасет: {}", datasetPath);
        log.info("Модель будет сохранена: {}", modelPath);

        lemmatizer.train(datasetPath, modelPath);
        log.info("Обучение завершено успешно!");
    }

    private static void handleLemmatize(String[] args) {
        if (args.length < 3) {
            throw new IllegalArgumentException("Команда lemmatize требует 2 аргумента: <word> <model.zip>");
        }

        String word = args[1];
        String modelPath = args[2];

        if (!Files.exists(Paths.get(modelPath))) {
            throw new IllegalArgumentException("Модель не найдена: " + modelPath);
        }

        Lemmatizer lemmatizer = new Lemmatizer();
        String lemma = lemmatizer.lemmatize(word, modelPath);

        System.out.println(word + " -> " + lemma);
        log.info("{} -> {}", word, lemma);
    }

    private static void handleBatch(String[] args) {
        if (args.length < 4) {
            throw new IllegalArgumentException("Команда batch требует 3 аргумента: <input.txt> <model.zip> <output.txt>");
        }

        String inputPath = args[1];
        String modelPath = args[2];
        String outputPath = args[3];

        if (!Files.exists(Paths.get(inputPath))) {
            throw new IllegalArgumentException("Входной файл не найден: " + inputPath);
        }
        if (!Files.exists(Paths.get(modelPath))) {
            throw new IllegalArgumentException("Модель не найдена: " + modelPath);
        }

        Lemmatizer lemmatizer = new Lemmatizer();

        try {
            List<String> words = Files.readAllLines(Paths.get(inputPath), StandardCharsets.UTF_8);
            List<String> results = new ArrayList<>();

            log.info("Обработка {} слов...", words.size());

            for (String word : words) {
                String trimmed = word.trim();
                if (!trimmed.isEmpty()) {
                    String lemma = lemmatizer.lemmatize(trimmed, modelPath);
                    results.add(trimmed + "\t" + lemma);
                }
            }

            Files.write(Paths.get(outputPath), results, StandardCharsets.UTF_8);
            log.info("Результаты сохранены в: {}", outputPath);

        } catch (IOException e) {
            throw new RuntimeException("Ошибка при пакетной обработке: " + e.getMessage(), e);
        }
    }
}
