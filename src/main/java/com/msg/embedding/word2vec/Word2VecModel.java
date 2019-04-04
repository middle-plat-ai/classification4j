package com.msg.embedding.word2vec;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;


@Slf4j
public class Word2VecModel {
    /**
     * word2vec包含两种结构:
     * 1) skip-gram结构:skip-gram结构是利用中间词预测邻近词
     * 2) cbow结构:cbow模型是利用上下文词预测中间词
     *
     * @param textPath  要训练词向量的分词后的文本
     * @param modelPath 要保存词向量模型的路径
     * @throws IOException
     */
    public static void train(String textPath, String modelPath) throws IOException {
        log.info("加载数据....");
        SentenceIterator iterator = new BasicLineIterator(new File(textPath));

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        log.info("构建模型....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)    //最小词频
                .batchSize(32)          //批大小
                .useAdaGrad(true)       //是否用ada
                .learningRate(0.001)    //学习率
                .iterations(10)         //迭代次数
                .layerSize(200)         //生成词向量维数
                .seed(42)               //随机数种子，为了复现结果
                .windowSize(5)          //窗口大小，前后多少个词作为训练输入或者输出
                .iterate(iterator)      //调用的数据
                .tokenizerFactory(tokenizerFactory)  //分词
                .build();
        log.info("开始训练模型....");
        vec.fit();

        log.info("模型写入到硬盘....");
        WordVectorSerializer.writeWord2VecModel(vec, modelPath);

    }

    public static Word2Vec load(String modelPath) {
        return WordVectorSerializer.readWord2VecModel(new File(modelPath));
    }


    public static void main(String[] args) throws IOException {
        String basicPath = Word2VecModel.class.getClassLoader().getResource("word2vec").getPath();
        System.out.println(basicPath);
        train(basicPath + "/words.txt", basicPath + "/words.bin");
        Word2Vec wordVectors = load(basicPath + "/words.bin");

        System.out.println(wordVectors.wordsNearest("人民", 10));
    }

}