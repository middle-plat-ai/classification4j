package com.msg.classifier.cnn;

import com.msg.classifier.util.TrainTestSplit;
import com.msg.classifier.util.FileUtil;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 用deeplearning4j训练cnn对文本进行分类的例子，很吃内存，设置8G以上，速度很慢，建议用GPU训练
 */
public class CnnTextClassifier {
    private String modelPath;
    private WordVectors wordVectors;
    private ComputationGraph net;
    private CnnSentenceDataSetIterator.Builder builder;

    private int batchSize = 32;                     //批次，以随机的32条数据作为一个批次进行更新权重
    private int vectorSize = 300;                   //词向量维度. Google News词向量模型的维度是300
    private int nEpochs = 1;                       //扫描一遍数据集为一个epoch，大小根据实际情况进行调整
    private int truncateReviewsToLength = 256;      //句子长度上限，即句子包含的最大单词数量
    private int cnnLayerFeatureMaps = 100;          //每种大小卷积核的数量
    private PoolingType globalPoolingType = PoolingType.MAX;        //采用max pooling的方式

    public CnnTextClassifier(String textPath, String vectorPath, String modelPath) {
        this.modelPath = modelPath;
        System.out.println("加载训练好的词向量：");
        wordVectors = WordVectorSerializer.loadStaticModel(new File(vectorPath));
        if ((net = loadTrainedModel()) == null) {
            net = train(textPath);
        }
        builder = buildCnnSentenceIterator();
    }

    public ComputationGraph train(String textPath) {
        Random rng = new Random(12345);                    //设置随机种子，使得每次运行程序都能获得同样的结果
        List<String> lines = FileUtil.readFileToArray(textPath);

        TrainTestSplit.TrainTest trainTest = TrainTestSplit.split(lines, 0.2f, "__label__", rng);

        //设置内存垃圾回收的周期为5s
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        //跟MultiLayerConfiguration类似，但是允许手动配置更多，有向无环图的构建方式
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .weightInit(WeightInit.RELU)                //权重参数初始化
                .activation(Activation.LEAKYRELU)           //激活函数
                .updater(new Adam(0.01))                      //权重更新方式
                .convolutionMode(ConvolutionMode.Same)      //卷积模式
                .l2(0.0001)            //正则化
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn1", new ConvolutionLayer.Builder()
                        .kernelSize(3, vectorSize)
                        .stride(1, vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn2", new ConvolutionLayer.Builder()
                        .kernelSize(4, vectorSize)
                        .stride(1, vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(5, vectorSize)
                        .stride(1, vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                //merge三种卷积核的卷积结果
                .addVertex("merge", new MergeVertex(), "cnn1", "cnn2", "cnn3")
                //池化操作
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(globalPoolingType)
                        .dropOut(0.5)
                        .build(), "merge")
                //用池化后得到的特征建立全连接层输出
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3 * cnnLayerFeatureMaps)
                        .nOut(2)    //最后输出两个类别，积极和消极
                        .build(), "globalPool")
                .setOutputs("out")
                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        System.out.println("输出每一层的参数值:");
        for (Layer l : net.getLayers()) {
            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
        }

        System.out.println("构建训练集和测试集：");

        DataSetIterator trainIter = getDataSetIterator(trainTest.getTrainX(), trainTest.getTrainY(), rng);
        DataSetIterator testIter = getDataSetIterator(trainTest.getTestX(), trainTest.getTestY(), rng);

        System.out.println("开始训练：");
        net.setListeners(new ScoreIterationListener(100));//每隔100个iteration就输出一次score，可视化的情况下可以去掉

        //可视化
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        net.setListeners(new StatsListener(statsStorage, 1));
//        uiServer.attach(statsStorage);

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            System.out.println("批次 " + i + " 完成，开始评估模型:");
            Evaluation evaluation = net.evaluate(testIter);
            System.out.println(evaluation.stats());
        }

        try {
            ModelSerializer.writeModel(net, new File(modelPath), true);
        } catch (IOException e) {
            System.out.println("写出到硬盘出错");
        }
        return net;
    }

    public String predict(String text) {
        LabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(new ArrayList<>(), new ArrayList<>(), new Random(12345));
        CnnSentenceDataSetIterator cnnSentenceDataSetIterator = builder.sentenceProvider(sentenceProvider).build();

        INDArray featuresFirstNegative = cnnSentenceDataSetIterator.loadSingleSentence(text);

        INDArray predictionsFirstNegative = net.outputSingle(featuresFirstNegative);

        //输出这个句子被模型预测为Negative和Positive的概率
        List<String> labels = new ArrayList<>();
        labels.add("no");
        labels.add("yes");
        System.out.println("\n第一个消极评论的预测:");
        int max = 0;
        double maxDouble = predictionsFirstNegative.getDouble(0);
        System.out.println("P(" + labels.get(0) + ") = " + predictionsFirstNegative.getDouble(0));
        for (int i = 1; i < labels.size(); i++) {
            double score = predictionsFirstNegative.getDouble(i);
            if (score >= maxDouble) {
                maxDouble = score;
                max = i;
            }
            System.out.println("P(" + labels.get(i) + ") = " + predictionsFirstNegative.getDouble(i));
        }
        return labels.get(max);
    }


    public ComputationGraph loadTrainedModel() {
        try {
            return ModelSerializer.restoreComputationGraph(new File(modelPath), true);
        } catch (IOException e) {
            return null;
        }
    }

    private CnnSentenceDataSetIterator.Builder buildCnnSentenceIterator() {
        return new CnnSentenceDataSetIterator.Builder(CnnSentenceDataSetIterator.Format.CNN2D)
                .wordVectors(wordVectors)
                .minibatchSize(batchSize)
                .maxSentenceLength(truncateReviewsToLength)
                .useNormalizedWordVectors(false);
    }


    private DataSetIterator getDataSetIterator(List<String> X, List<String> Y, Random rng) {
        LabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(X, Y, rng);
        return builder.sentenceProvider(sentenceProvider).build();
    }

    public static void main(String[] args) {

        String textPath = "/home/msg/workspace/idea/machine-learning-tutorials/deeplearning4j-tutorials/src/main/resources/data/train.txt";
        String vectorPath = "/home/msg/Documents/notebook/fasttext/cc.id.300.vec";

        String modelPath = "/home/msg/workspace/idea/machine-learning-tutorials/deeplearning4j-tutorials/src/main/resources/cnn/model";

//        CnnTextClassifier cnnTextClassifier = new CnnTextClassifier(textPath, vectorPath, modelPath);
//
//        cnnTextClassifier.predict("hello world");

        String label = new CnnTextClassifier(textPath, vectorPath, modelPath).predict("mau langganan nsp despacito remix luis fonsi daddy yankee justin bieber tadi bls ya menangkan unit honda brio yamaha aerox trf");

        System.out.println(label);

    }
}
