import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CNNModel {
    public static void main(String[] args) throws IOException, InterruptedException {
        long seed = 1234;
        double learningRate = 0.001;
        long height = 28;
        long width = 28;
        long depth=1; //Quand image est en noir et blanc : la profondeur =1 (en cas de en couleur égale 3)
        int outputSize=10;  //nbr de nombres

        /* Configuration du modèle  */
        System.out.println("----------Configuration--------");
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
        .seed(seed)   //Pour génerer le m nbr aléatoire chaque fois qu'on éxécute
        .updater(new Adam(learningRate))
        .list()        //liste de couches qu'on va créer
        .setInputType(InputType.convolutionalFlat(height, width, depth))  //Spécifier la type d'entrée (dans ce exemple : image) et après donner ces tailles
        /* Création des couches */
        .layer(0, new ConvolutionLayer.Builder()
                .nIn(depth)    //nbr de input (1image)
                .nOut(20)      //Après le filtre on obtienne 20 images
                .activation(Activation.RELU) //on utilise RELU comme fonction d'activation (si nbr négative, la fonction lui transforme en 0)
                .kernelSize(5, 5)   //5x5 cad la dimension de filtre
                .stride(1, 1)  //le filtre doit etre glisser horizontalement par 1 et verticalement par 1 sur l'image
                .build())
        .layer(1, new SubsamplingLayer.Builder()
                .kernelSize(2,2)
                .stride(2, 2)
                .poolingType(SubsamplingLayer.PoolingType.MAX)  //Pour ce filtre il prend le max de chaque 4 carré (image)
                .build())
        .layer(2, new ConvolutionLayer.Builder()
                .nOut(50)
                .activation(Activation.RELU)
                .kernelSize(5,5)
                .stride(1,1)
                .build())
        .layer(3, new SubsamplingLayer.Builder()
                .kernelSize(2,2)
                .stride(2, 2)
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .build())
        .layer(4, new DenseLayer.Builder()
                .nOut(500)
                .activation(Activation.RELU)
                .build())
        .layer(5, new OutputLayer.Builder()
                .nOut(outputSize)
                .activation(Activation.SOFTMAX)  //calcule la probabilité de nombre (ex: 0.9 que le nbr est 3)
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)  //Pour minimiser l'erreur
                .build())
        .build();


        /* Création du Modèle */
        System.out.println("----------Création du modèle---------");
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        /* Entrainement du modèle */
        System.out.println("----------Model training...----------");

        String path = System.getProperty("user.home")+"\\Desktop\\Emsi\\Complements de formation\\2Machines et Deep Learning for Java Application with DL4j\\mnist_png";
        File fileTrain = new File(path+"/training");
        FileSplit fileSplitTrain = new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));  //ALLOWED_FORMATS: Lire n'importe quel format d'image (png, jpeg ...)
        RecordReader recordReaderTrain = new ImageRecordReader(height, width, depth, new ParentPathLabelGenerator());    //Permet de lire les images
        recordReaderTrain.initialize(fileSplitTrain);    //RecordReader lire les doonées à partir de fileSpitTrain
        int batchSize = 54;
        /* BatchSize? Par exemple, supposons que vous ayez 1050 échantillons d'apprentissage et que vous souhaitiez
        en définir un batch_sizeégal à 100. L'algorithme prend les 100 premiers échantillons (du 1er au 100e) de
        l'ensemble de données d'apprentissage et entraîne le réseau. Ensuite, il prend les 100 deuxièmes échantillons
        (du 101e au 200e) et entraîne à nouveau le réseau. Nous pouvons continuer à faire cette procédure jusqu'à ce
        que nous ayons propagé tous les échantillons à travers le réseau. Un problème peut survenir avec la dernière série d'échantillons.
         */
        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, outputSize);
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);            //Normalisation des données (chaque image doit la ramnener entre 0 et 1)
        dataSetIteratorTrain.setPreProcessor(scaler);
        /* Affichage des résultats (output)  de images entrés (input)  */
      /*  while(dataSetIteratorTrain.hasNext()){
            DataSet dataSet = dataSetIteratorTrain.next();
            INDArray features = dataSet.getFeatures();  // //Feature = Input = Image (ex : [54,1,28,28] = Pour chaque batch on a 54 images, et pour chaque image qui est en noir : profond=1  & sa dimension=28x28 )
            INDArray labels = dataSet.getLabels();       // label = Output
            System.out.println(features.shapeInfoToString());
            System.out.println(labels);
            System.out.println("--------------------------");
        }  **/

        /* Serveur de l'affichage de l'entrainement de notre modele */
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));


        int numEpoch = 1; //car on a un grnd nbr d'images qui va entrainer notre modèle
        //Entrainement du modèle
        for (int i = 0; i < numEpoch ; i++) {
            model.fit(dataSetIteratorTrain);
        }

        /* Evaluer le modèle en lui donnant les données de test */
        System.out.println("---------Model Evaluation:----------");
        File fileTest = new File(path+"/testing");
        FileSplit fileSplitTest= new FileSplit(fileTest, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RecordReader recordReaderTest= new ImageRecordReader(height, width, depth, new ParentPathLabelGenerator());
        recordReaderTest.initialize(fileSplitTest);

        DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, outputSize);
        DataNormalization scalerTest = new ImagePreProcessingScaler(0,1);
        dataSetIteratorTrain.setPreProcessor(scalerTest);

        Evaluation evaluation = new Evaluation();
        while(dataSetIteratorTest.hasNext()){
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features = dataSet.getFeatures();   //Feature = Input = Image (ex : [54,1,28,28] = Pour chaque batch on a 54 images, et pour chaque image qui est en noir : profond=1  & sa dimension=28x28 )
            INDArray targetLabels = dataSet.getLabels();   // Résultats des images (cad est-ce que l'image présente 0 ou 1 ou 2 ...)

            //Model qui va faire prédiction des images
            INDArray predicted = model.output(features);
            //Après on compare targetLabels(résultat 100% vrai) et predicted(les prédictions effectués par le modèle)
            evaluation.eval(predicted, targetLabels);
        }
        System.out.println(evaluation.stats());


        /* Enregistrer le modèle  */
        ModelSerializer.writeModel(model, new File("model.zip"), true);

    }
}
