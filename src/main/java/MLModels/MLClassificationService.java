package MLModels;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import java.util.stream.Stream;

public class MLClassificationService {


    public JavaSparkContext createSparkContext() {
        SparkConf conf = new SparkConf().setAppName("ML")
                .setMaster("local");

        return new JavaSparkContext(conf);
    }



    public LogisticRegressionModel dataSplitAndModelCreationAndAccuracy(JavaRDD<LabeledPoint> parsedData) {

        // 5. Data Splitting into 80% Training and 20% Test Sets
        JavaRDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[] { 0.1, 0.9 }, 11L);
        JavaRDD<LabeledPoint> trainingData = splits[0].cache();
        JavaRDD<LabeledPoint> testData = splits[1];

        RDD<LabeledPoint> rdd = trainingData.rdd();

        System.out.println("Model Accuracy on Test Data: " + "im hereee");

        LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(rdd);

        // 6.2. Model Evaluation
        JavaPairRDD<Object, Object> predictionAndLabels = testData.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double accuracy = metrics.accuracy();
        System.out.println("Model Accuracy on Test Data: " + accuracy);
        return model;
    }

    public void modelSaving(LogisticRegressionModel model, JavaSparkContext sc, String modelSavePath) {
        model.save(sc.sc(), modelSavePath);
    }

    public LogisticRegressionModel loadModel(JavaSparkContext sc, String modelSavePath) {
        LogisticRegressionModel model = LogisticRegressionModel.load(sc.sc(), modelSavePath);
        return model;
    }

    public int newDataPrediction(LogisticRegressionModel model, double[] testData) {


        System.out.println("Prediction label for new data given : "+"im here");

        Vector newData = Vectors.dense(testData);
        double prediction = model.predict( newData);
        System.out.println("Prediction label for new data given : " + prediction);
        return (int)prediction;
    }

    protected static JavaRDD<LabeledPoint> loadDataFromFileAndDataPreparation(JavaSparkContext sc, String inputFile) throws IOException {
        File file = new File(inputFile);
        JavaRDD<String> data = sc.textFile(file.getPath());

        // Removing the header from CSV file
        String header  = data.first();

        data = data.filter(line ->  !line.equals(header));


        System.out.println("Done Until Now !!!!!!!!!!!!!!");

        return data.
                map(line -> {
                    System.out.println(line);
                    line = line
                           .replace("Teacher", "1")
                            .replace("Engineer", "2")
                            .replace("Nurse", "3")
                            .replace("Doctor", "4")
                            .replace("Businessman", "5")
                            .replace("Salesperson","6")
                            .replace("Entrepreneur","7")
                            .replace("Lawyer","8")
                            .replace("Accountant","9")
                            .replace("Software Dev","10")
                            .replace("Electrician","11")
                            .replace("Architect","12")
                            .replace("Marketing Exec","13")
                            .replace("Chef","14")
                            .replace("Police Officer","15")
                            .replace("Real Estate Ag","16")
                            .replace("Pilot","17")
                            .replace("Graphic Design","18")
                            .replace("Writer","19")
                            .replace("Musician","20")
                            .replace("10eloper","21");

                    String[] split = line.split(",");

                    double[] featureValues = Stream.of(split)
                            .mapToDouble(Double::parseDouble).toArray();

                    if (featureValues.length > 3) {

                        double label = featureValues[3];

                        featureValues = Arrays.copyOfRange(featureValues,0, 3);

                        return new LabeledPoint(label, Vectors.dense(featureValues));
                    }
                    System.out.println("Im Here");
                    return null;
                }).cache();
    }


}
