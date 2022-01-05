import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Image 
{
   String folder = "C:\\Users\\arnav\\eclipse-workspace\\ATCS NNs\\src\\activation files\\";
   String fileName;
   String fullname;
   String name;
   double[] values;
   int imgLen = 4800;
   double truth;
   
   
   public Image(String fname)
   {
      fullname = fname;
      fileName = folder + fname;
      name = fname.substring(0, 2);
      
      values = new double[imgLen];
   }
   
   public void calcTruthTrain()
   {
      String truString = name.substring(0,1);
      truth = Double.valueOf(truString);
   }
   
   public void calcTruthTest()
   {
      String truString = name.substring(1,2);
      truth = Double.valueOf(truString);
   }
   
   public double getTruth()
   {
      return truth;
   }
   
   public void populateValues()
   {
      File imgFile = new File(fileName);
      Scanner sc = null;
      
      
      try 
      {
         sc = new Scanner(imgFile);
      } 
      catch (FileNotFoundException e) 
      {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
      
      for (int i = 0; i < imgLen; i++)
      {
         values[i] = sc.nextDouble();
      }
      
      sc.close();
   }
   
   public double[] getArray()
   {
      return values;
   }
   
   public String getName()
   {
      return name;
   }
   

   public static void main(String[] args) 
   {
      // TODO Auto-generated method stub

   }

}
