package com.msg.classifier.util;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.*;

/**
 * Created by MSG on 2018-11-01:上午9:50
 */
public class TrainTestSplit {

    /**
     * 拆分后的数据及标签对象
     */
    @Data
    public static class TrainTest {
        private List<String> trainX;
        private List<String> testX;
        private List<String> trainY;
        private List<String> testY;

        private Set<String> labels;
        private List<String> wrongTexts;
    }

    /**
     * 拆分数据集，拆分成四部分，trainX，testX，trainY，testY
     *
     * @param lines       待拆分数据集, 格式为用labelSplit拆分为两部分,第一部分为分好词的数据,第二部分为标签,参考fasttext数据格式
     * @param testPercent 测试集的百分比
     * @param labelSplit  拆分数据和标签的拆分符
     * @return
     */
    public static TrainTest split(List<String> lines, float testPercent, String labelSplit, Random rng) {
        TrainTest trainTest = new TrainTest();

        List<String> wrongTexts = new ArrayList<>();
        Map<String, List<String>> labelMap = new HashMap<>();

        for (String line : lines) {
            String[] lineArr = line.split(labelSplit);
            String label = lineArr[1];
            if (label == null || "".equals(label.trim())) {
                wrongTexts.add(line);
                continue;
            }
            label = label.trim();

            List<String> labelList = labelMap.computeIfAbsent(label, k -> new ArrayList<>());
            labelList.add(lineArr[0].trim());
        }

        trainTest.setWrongTexts(wrongTexts);
        trainTest.setLabels(labelMap.keySet());

        List<String> trainX = new ArrayList<>();
        List<String> trainY = new ArrayList<>();
        List<String> testX = new ArrayList<>();
        List<String> testY = new ArrayList<>();

        for (Map.Entry<String, List<String>> entry : labelMap.entrySet()) {
            String label = entry.getKey();
            List<String> labelList = entry.getValue();
            int numAll = (int) (testPercent * (labelList.size() - 1)) + 1;
            Collections.shuffle(labelList, rng);
            for (int i = numAll; i < labelList.size(); i++) {
                String line = labelList.get(i);
                trainX.add(line);
                trainY.add(label);
            }

            for (int i = 0; i < numAll; i++) {
                String line = labelList.get(i);
                testX.add(line);
                testY.add(label);
            }
        }
        trainTest.setTrainX(trainX);
        trainTest.setTrainY(trainY);
        trainTest.setTestX(testX);
        trainTest.setTestY(testY);

        System.out.println("训练集和测试集的拆分情况：");

        System.out.println("\ttrainX size:" + trainX.size());
        System.out.println("\ttrainY size:" + trainY.size());
        System.out.println("\ttestX size:" + testX.size());
        System.out.println("\ttestY size:" + testY.size());

        System.out.println("\twrong size:" + wrongTexts.size());
        System.out.println("\twrong text:" + wrongTexts);
        return trainTest;
    }
}
