package org.example;

import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Точка входа приложения.
 */
@Slf4j
public class LemmatizerStrarter {

    public static void main(String[] args) {
        log.info("LemmatizerStrarter started with args: {}", Arrays.toString(args));

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
                .epochs(5)
                .batchSize(64)
                .lstmSize(128)
                .learningRate(0.0005)
                .build();

        if (!Files.exists(Paths.get(datasetPath))) {
            throw new IllegalArgumentException("Датасет не найден: " + datasetPath);
        }

        log.info("Начало обучения модели...");
        log.info("Датасет: {}", datasetPath);
        log.info("Модель будет сохранена: {}", modelPath);

        new LemmatizerModelTrainer(config).train(datasetPath, modelPath);
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

        String lemma = new Lemmatizer().lemmatize(word, modelPath);
        System.out.println(word + " -> " + lemma);
        log.info("{} -> {}", word, lemma);
    }
}
