import MLModels.SparkMLUnit;

import java.util.Arrays;

public class Runner  {

    public static void main(String[] args) {
      System.out.println("System is Running Now . . .");

      SparkMLUnit mlUnit = new SparkMLUnit();

      double[] testData = new double[] {20.0,2500.0,760};

      try {
          System.out.println("System Started Now");

          mlUnit.machineRunner(testData);

          System.out.println("System Finished Now");

      }catch (Exception exception){

          System.out.println(exception.getMessage());

          System.out.println("Error Happen !");
      }

    }


}
