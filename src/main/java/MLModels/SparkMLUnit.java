package MLModels;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.IOException;

import static MLModels.MLClassificationService.loadDataFromFileAndDataPreparation;

public class SparkMLUnit {

    public void machineRunner(double[] testData) throws IOException {

        MLClassificationService demo = new MLClassificationService();
        JavaSparkContext sc = demo.createSparkContext();

        // Data preparation
        String inputFile = "src/main/resources/Files/mydata.csv";

        JavaRDD<LabeledPoint> parsedData = loadDataFromFileAndDataPreparation(sc, inputFile);
        LogisticRegressionModel model = demo.dataSplitAndModelCreationAndAccuracy(parsedData);

        /*
         * //Saving and Retrieval model String modelSavePath =
         * "model\\logistic-regression"; demo.modelSaving(model, sc, modelSavePath);
         * model = demo.loadModel(sc, modelSavePath);
         */

        demo.newDataPrediction(model, testData);
        // Close Spark Context
        sc.close();

    }




}
