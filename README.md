## 相关NLP算法(Java实现)

### 1) 相关包依赖

```xml
        <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-cuda-9.0</artifactId>
            <version>${dl4j.version}</version>
        </dependency>
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-cuda-9.0</artifactId>
            <version>${dl4j_cuda.version}</version>
        </dependency>

```

 * 采用上述jar包,不论采用何种模型,训练的时候,应该会在控制台输出GPU相关信息,证明正在用GPU进行训练,若是没有GPU或者GPU不可用,则需要切换nd4j版本和deeplearning4j版本.
 
 ```xml
         <dependency>
             <groupId>org.deeplearning4j</groupId>
             <artifactId>deeplearning4j-core</artifactId>
             <version>${dl4j.version}</version>
         </dependency>
         <dependency>
             <groupId>org.nd4j</groupId>
             <artifactId>nd4j-native-platform</artifactId>
             <version>${dl4j.version}</version>
         </dependency>
 ```

* 其它依赖包详见pom.xml文件.

### 2) 训练word2vec词向量

```java
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
```

* 代码如上所示.设置好相关参数,即可进行训练,其中使用的时候,示例代码如下所示:

```java
    String basicPath = Word2VecModel.class.getClassLoader().getResource("word2vec").getPath();
    System.out.println(basicPath);
    Word2VecModel.train(basicPath + "/words.txt",basicPath+"/words.bin");
    WordVectors wordVectors = load(basicPath + "/words.bin");
    System.out.println(wordVectors.wordsNearest("人民",10));
```

### 3) CNN算法实现文本分类

```java
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
```

* 构建一个CNN网络,然后训练.如果想要可视化训练过程,加入以下代码:

```java
    //可视化
    UIServer uiServer = UIServer.getInstance();
    StatsStorage statsStorage = new InMemoryStatsStorage();
    net.setListeners(new StatsListener(statsStorage, 1));
    uiServer.attach(statsStorage);
```

* 目前只做了二分类,而且推理的时候,分类直接写在了代码里,工程中可以考虑抽取出来,将类别写入文件里.

