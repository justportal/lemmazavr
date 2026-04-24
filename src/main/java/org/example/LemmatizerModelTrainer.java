package org.example;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.GradientNormalization;
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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

/**
 * Класс для обучения модели лемматизации.
 */
@Slf4j
public class LemmatizerModelTrainer {

    private final LemmatizerConfig config;
    private LemmatizerVocabulary vocabulary;

    public LemmatizerModelTrainer() {
        this(null);
    }

    public LemmatizerModelTrainer(LemmatizerConfig config) {
        this.config = config;
        if (config != null) {
            config.validate();
        }
    }

    /**
     * Обучает модель на датасете.
     *
     * @param datasetPath путь к TSV файлу (словоформа\tлемма)
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
            vocabulary = LemmatizerVocabulary.buildFromDataset(dataset);
            long sampleCount = vocabulary.getSampleCount();

            if (sampleCount == 0) {
                throw new IllegalStateException("Датасет пуст или не содержит валидных строк.");
            }

            int batchCount = (int) Math.ceil((double) sampleCount / trainConfig.getBatchSize());
            log.info("Словарь построен. Размер: {}, MAX_LEN: {}, примеров: {}, батчей на эпоху: {}",
                    vocabulary.getVocabSize(), trainConfig.getMaxLen(), sampleCount, batchCount);
            log.info("Gradient clipping: {} with threshold {}",
                    GradientNormalization.ClipElementWiseAbsoluteValue, trainConfig.getGradientClipThreshold());

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
                        vocabulary.getCharToIndex(),
                        vocabulary.getVocabSize()
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

    private MultiLayerNetwork buildModel(LemmatizerConfig cfg) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(cfg.getSeed())
                .updater(new Adam(cfg.getLearningRate()))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new LSTM.Builder()
                        .nIn(vocabulary.getVocabSize())
                        .nOut(cfg.getLstmSize())
                        .activation(Activation.TANH)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(cfg.getGradientClipThreshold())
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(cfg.getLstmSize())
                        .nOut(cfg.getLstmSize())
                        .activation(Activation.RELU)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(cfg.getGradientClipThreshold())
                        .build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(cfg.getLstmSize())
                        .nOut(vocabulary.getVocabSize())
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(cfg.getGradientClipThreshold())
                        .build())
                .build();

        return new MultiLayerNetwork(conf);
    }

    private void saveModel(MultiLayerNetwork model, String outputPath) throws IOException {
        File modelFile = new File(outputPath);
        File parent = modelFile.getParentFile();

        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        ModelSerializer.writeModel(model, modelFile, true);
        ModelSerializer.addObjectToFile(modelFile, LemmatizerVocabulary.VOCABULARY_METADATA_KEY, vocabulary.toMetadata());
    }
}
