import java.util.Arrays;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;

/**
 * 
 * @author Arnav Dani
 * 
 * Created: 10-11-2021
 * 
 * Simulates the behavior of an A-B-C neural network that has one hidden layer
 * The hidden layer, input layer, and output layer can have any number of activations
 * 
 * 
 * The weights are optimized using a gradient descent and back propagation is utilized
 * to make the process as efficient as possible.
 * 
 * In addition, file reading and writing is implemented.
 * 
 * 
 * Inputs and truth table are hardcoded in
 *
 */

public class Network 
{
   double[] outputs;       //output activation
   double[][] inputs;      //input activations for xor
   double[] hiddens;       //nodes in the 1 hidden layer
   double[][][] weights;   //weights in the network
   double[][] truthtable;  //truth table for xor
   String[] outNames;      //clarifies the output being evaluated (Or/and/xor etc)
   double[][] errorVals;   //store errors from each pass to use for error checks
   int n_inputs;           //constant with the number of sets of inputs, 4 for xor
   int n_hiddens;          //constant for the number of hidden nodes
   int n_layers;           //number of layers
   int n_outputs;          //number of outputs
   double start;           //start of random range
   double end;             //end of random range
   int inputSetSize;       //size of training set
   double maxError;        //largest error from a single pass
   double totalError;      //sum of all errors from a single pass
   
   
   
   /*
    * exit condition variables
    */
   
   int n_iters;                  //max number of iterations the network should pass through
   final int N_ERRORCHECKS = 3;
   int curr_iters;               //current number of iterations
   double lambda;                //learning rate
   boolean[] exitConditions;
   
   boolean repeat;               //says whether to do another training pass
   boolean train;                //says whether the network is training or running
   boolean preload;              //whether to use preloaded weights or fixed weights
   
   int dimCount;
   
   /*
    * training variables
    */
   double[] omega;
   double[] thetai;
   double[] thetaj;        
   double[] psiLowerI;
   double[][] deltaWeightsJ;
   double e_thresh;        //error threshold for an individual case
   double tot_thresh;      //total combined error for all cases
   
   double omegaJ;
   double upperPsiJ;
   double[][] deltaWeightsK;
   
   /*
    * calculating runtime
    */
   long startTime;
   long endTime;
   long elapsed;
   
   /*
    * file reading/writing
    */
   
   File inputFile;
   FileWriter outputFile;
   
   
      
   /**
    * Constructor for the network class
    * calls the method that sets up all the fundamental arrays 
    * and initializes the output variable
    * 
    * @param inFile input file to read config from
    * @param outFile the output file to write results to
    */
   public Network(File inFile, FileWriter outFile)
   {     
      inputFile = inFile;
      outputFile = outFile;
      readfile();
   } 
   
   /*
    * Determines whether the program is training or testing
    * @return true if training; false if testing
    */
   public boolean isTraining()
   {
      return train;
   }
   
   /**
    * Initializes all the arrays, sets the hard coded weights, and updates all the error checks
    * 
    */
   public void configureNetwork()
   {  
      if (train)
      {
         thetaj = new double[n_hiddens];
         thetai = new double[n_outputs];
         deltaWeightsJ = new double[n_hiddens][n_outputs];
         deltaWeightsK = new double[n_inputs][n_hiddens];
         psiLowerI = new double[n_outputs];
         omega = new double[n_outputs];
      } //if (train)
      
      n_layers = 2;                          //excluding output layer since no connection leaves the output layer
      hiddens = new double[n_hiddens];
      outputs = new double[n_outputs];
     
      
      dimCount = Math.max(n_inputs, n_hiddens); 
      dimCount = Math.max(dimCount, n_outputs);           //to ensure all connections are represented
      
      weights = new double[n_layers][dimCount][dimCount]; //weights are between layers A-B and B-C
      
      
      if (!preload)
      {
         randomizeWeights();
      }
         
      resetNetwork();
   } //public void configureNetwork()

   
   /**
    * identifies the type of targets
    */
   public void setTargets()
   {
      outNames = new String[n_outputs];
      outNames[0] = "OR";
      outNames[1] = "AND";
      outNames[2] = "XOR";
     
   } //public void setTargets()
      
   /**
    * resets all mutable instance variables to their initial, starting value
    * 
    * Currently, this does minimal work but as more variables get added, there
    *       will be more to reset
    */
   public void resetNetwork()
   {
      exitConditions = new boolean[4];
      curr_iters = 0;
      totalError = 0;
      maxError = 0;
      repeat = true;
      
      for (int i = 0; i < N_ERRORCHECKS; i++)
      {
         exitConditions[i] = false;
      }
   } //public void resetNetwork()
   
   /**
    * Activation function
    * 
    * This describes the sigmoid function being used
    * 
    * This can easily be replaced with any other function, but the derivative will have
    * to be updated as well; the method for the derivative is lower
    * 
    * @param x input into the function
    * @return type double of x passed through the function
    */
   public double activate(double x)
   {
      return  1.0 / (1.0 + Math.exp(-x));
   }
   
   /**
    * derivative of the activation function
    * 
    * the current activation function is sigmoid: the derivative is f(1-f)
    * 
    * @param x the input value to pass through the derivative
    * @return the value after being passed through the function 
    */
   public double actDeriv(double x)
   {
      double act = activate(x);
      return act * (1.0 - act);
   }
   
   /**
    * calculates the value of the activations in the hidden layers
    * using dot products of the weights of the input layers
    * 
    * uses variable val to avoid += from messing up values in the hidden layer
    * @param num refers to the input type
    */
   public void calcHiddens(int num)
   {
      double val = 0.0;
      for (int j = 0; j < n_hiddens; j++)
      {
         val = 0.0;    
         
         for (int k = 0; k < n_inputs; k++)
         {
            val += inputs[num][k] * weights[0][k][j];
         }
         
         hiddens[j] = activate(val);
         
      } //for (int j = 0; j < n_hiddens; j++)
   } //public void calcHiddens(int num)
   
   
   
   /**
    * Used for training
    * 
    * calculates the value of the activations in the hidden layers
    * using dot products of the weights of the input layers
    * 
    * uses variable val to avoid += from messing up values in the hidden layer
    * 
    * stores information for training
    * @param num refers to the input type
    */
   public void trainCalcHiddens(int num)
   {
      double val = 0.0;
      for (int j = 0; j < n_hiddens; j++)
      {
         val = 0.0;    
         
         for (int k = 0; k < n_inputs; k++)
         {
            val += inputs[num][k] * weights[0][k][j];
         } //for (int k = 0; k < n_inputs; k++)
         
         hiddens[j] = activate(val);
         
         thetaj[j] = val;
         
      } //for (int j = 0; j < n_hiddens; j++)
   } //public void trainCalcHiddens(int num)
   
   
   /**
    * calculates the final output and theta(i) values using the dot products of the weights
    * in the hidden layer
    * 
    * @param input identifies the binary input being used
    * 
    */
   public void calcOutput(int input)
   {
      
      for (int i = 0; i < n_outputs; i++)
      {
         double val = 0.0;
         
         for (int j = 0; j < n_hiddens; j++)
         {
            val += hiddens[j] * weights[1][j][i];
         }
         
         outputs[i] = activate(val);
         
      } //for (int i = 0; i < n_outputs; i++)
   } //public void calcOutput(int input)
   
   
   
   /**
    * used for training
    * 
    * calculates the final output and theta(i) values using the dot products of the weights
    * in the hidden layer
    * 
    * calculates many values used in training
    * 
    * @param input identifies the binary input being used
    * 
    */
   public void trainCalcOutput(int input)
   {
      
      for (int i = 0; i < n_outputs; i++)
      {
         double val = 0.0;
         
         for (int j = 0; j < n_hiddens; j++)
         {
            val += hiddens[j] * weights[1][j][i];
         }
         
         outputs[i] = activate(val);
         
         thetai[i] = val;
         omega[i] = getError(i, input);
         errorVals[input][i] = omega[i];
         psiLowerI[i] = omega[i] * actDeriv(thetai[i]);        
         
      } //for (int i = 0; i < n_outputs; i++)
   } //public void trainCalcOutput(int input)
   
   
   
   /**
    * 
    * Since there are 4 possible inputs for an Xor, n represents
    * which input is being tested to that the output of the network
    * can be correctly compared to the correct ground truth
    * 
    * 
    * @param n the row in the truth table being evaluated
    * @param out represents each output and its unique truth table
    * @return double representing the error after passed through the function
    */
   public double getError(int out, int num)
   {
      double error = truthtable[num][out] - outputs[out];
      return error;
   }
   
   /**
    * combines the methods defined to perform a forward pass through the network
    * @param the input being considered [(0, 1), (1, 1), etc]
    */
   public void forwardPass(int num)
   {
      calcHiddens(num);
      calcOutput(num);
   }
   
   /**
    * combines the methods defined to perform a forward pass through the network
    * @param the input being considered [(0, 1), (1, 1), etc]
    */
   public void trainForwardPass(int num)
   {
      trainCalcHiddens(num);
      trainCalcOutput(num);
   }
   
   /**
    * runs the network by letting the user set the inputs being used
    * and passing the inputs forward through the network
    * 
    */
   public void run()
   {
      /*
       * give values to the weights and initializes the inputs
       * being passed in through the network
       */
      
      
      setTargets(); //helps split between the 3 output cases
      
      for (int i = 0; i < inputSetSize; i++)
      {
         System.out.println("\n\n");
         forwardPass(i);
         displayRunResults(i);
      }
       //for (int i = 0; i < n_outputs; i++)
   } //public void run()

   /*
    * training specific code
    * 
    * The code below this point includes randomly setting and initializing weights
    * calculates the change in weights
    * checks for exit
    * 
    */
   
   /**
    * Random number generator that returns a random double within a specific range
    * 
    * @param start start of the range of numbers to pick a random number from
    * @param end end of the range to pick a random number from
    * @return random integer between start and end
    */
   public double randomgen(double start, double end)
   {
      double diff = end - start;
      double rand = start + Math.random()*diff;
      
      return rand;
   }
   
   /**
    * assigns all the relevant weights to random values
    * 
    * the random range is defined by parameters passed in through the constructor
    */
   public void randomizeWeights()
   {
      for (int n = 0; n < n_layers; n++)
      {
         for (int k = 0; k < dimCount; k++)
         {
            for (int j = 0; j < dimCount; j++)
            {
               weights[n][k][j] = randomgen(start, end); 
            }
         } //for (int k = 0; k < n_inputs; k++)
      } // for (int n = 0; n < n_layers; n++)
      
   } //public void randomizeWeights()
   
   
   /**
    * Implements gradient descent to calculate the optimal change for each
    * each and stores that amount to apply later
    * 
    * weight changes must be applied later to stay consistent with the mathematical formulas
    * 
    * @param input identifies the input set - the integer helps identify whether the input was 
    *    0,0     0,1     1,0     1,1
    * 
    */
   public void calculateWeights(int input)
   {
      trainForwardPass(input);           //have to do an initial forward pass to do calculations
      
      for (int j = 0; j < n_hiddens; j++)
      {
         omegaJ = 0.0;    
         for (int i = 0; i < n_outputs; i++)
         {
            omegaJ += psiLowerI[i] * weights[1][j][i];            
            weights[1][j][i] += lambda * hiddens[j] * psiLowerI[i];
         } //for (int i = 0; i < n_outputs; i++)
         
         upperPsiJ = omegaJ * actDeriv(thetaj[j]);
         
         for (int k = 0; k < n_inputs; k++)
         {                     
            weights[0][k][j] += lambda * inputs[input][k] * upperPsiJ;
         } //for (int k = 0; k < n_inputs; k++)
         
      } //for (int j = 0; j < n_hiddens; j++)
      
   } //public void calculateWeights(int input)   
      
   
   /**
    * checks whether all and any exit conditions are met
    * updates the array that stores all the exit conditions
    * 
    * When any condition is true, it changes a value in the boolean array
    * from false to true. This way, all conditions that are met can be
    * kept track of.
    * 
    * Method also returns whether any condition has been met as a boolean check
    * prior to executing exit procedure
    * 
    * @return true if any exit condition is met; otherwise, false
    */
   public boolean checkExit()
   {
      /*
       * check 1 - if lambda is zero
       */
      if (lambda == 0)
            exitConditions[0] = true;
      
      /*
       * check 2 - if the number of iterations hits the max
       */
      if (curr_iters >= n_iters)
            exitConditions[1] = true;     
      
      /*
       * check 3 - error thresholding
       * 
       * The error check I use has 2 steps
       * 
       * First - every error value must be less than the absolute value of an error threshold
       * for all 4 inputs - for example, if e_thresh is 0.05, the error must be <0.05 or <-0.05
       * 
       * Second - the sum of the absolute values of all errors should be less than 2 times the threshold (tot_thresh)
       * 
       * Both are necessary - only checking total error could allow outliers to pass through
       * and only checking individual error requires a higher threshold to precision
       * 
       * The thresholds are mutable via a parameter passed through the constructor
       */
      double currentError = 0.0;
      boolean errcheck = true;
      maxError = 0.0;
      totalError = 0.0;
      
      for (int i = 0; i < inputSetSize; i++)
      {
         for (int j = 0; j < n_outputs; j++)
         {
            currentError = errorVals[i][j];
            totalError += Math.abs(currentError);
            if (currentError > e_thresh || currentError < -e_thresh )
            {           
               errcheck = false;
            }   
            else
            {
               maxError = Math.max(maxError, Math.abs(currentError));
            }
           
         } //for (int j = 0; j < inputSetSize; j++)
      } //for (int i = 0; i < n_outputs; i++)
      
      
      if (totalError < tot_thresh || totalError < -tot_thresh)
      {
         exitConditions[2] = errcheck;
      }
      
      /*
       * as more error checks are added, they will be processed here
       */
      
      /*
       * if there is one true error check, return true to exit the method
       */
      
      boolean exit = false;
      
      for (int i = 0; i < N_ERRORCHECKS; i++)
      {
         if (exitConditions[i] == true)
               exit = true;
      }
      
      return exit;
   } //public boolean checkExit()
   
   /**
    * exits the training loop by changing the repeat boolean to false
    * 
    * This signifies that the loop should no longer be repeating
    * 
    */
   public void exit()
   {
      repeat = false;
   }
   
   /**
    * executes the full training sequence to train the network
    * to be able to recognize an XOR pattern
    * 
    */
   public void train()
   {     
      resetNetwork();
      setTargets();
      
      startTime = System.currentTimeMillis();
      
      while (repeat)
      {  
         calculateWeights(curr_iters % inputSetSize);
         
         curr_iters++;
         
         if (checkExit())
         {
            exit();
         }
            
      } //while (repeat)
      
      endTime = System.currentTimeMillis();
      elapsed = endTime - startTime;
      
      finishTraining();
      
   } //public void train()
   
   /*
    * getting results and printing
    * 
    * Lists all the reasons why the training sequence terminated and 
    * then prints information about the network and results
    */
   
   public void finishTraining()
   {  
      
      displayNetworkConfig();
      
      System.out.println("Training done\nReason for finishing:");
      
      if (exitConditions[0] == true)
         System.out.println("Lambda became 0");
      
      
      if (exitConditions[1] == true)
         System.out.println("Max number of iterations of " + n_iters + " approached");
      
      if (exitConditions[2] == true)
         System.out.println("Error Threshold met - the highest magnitude of error for an individual case was " +
                     maxError + " which is less than " + 
                     e_thresh + "\n\t\tCombined error was " + totalError + " which is less than " + tot_thresh +
                     "\n\nBoth of these conditions had to be met for termination");
      
      
      System.out.println("\n\nNumber of iterations at ending: " + curr_iters);
      
      /*
       * add more prints for each added exit condition
       */
      
      
      /*
       * last forward pass for results
       */
      
      for (int i = 0; i < inputSetSize; i++)
      {
            trainForwardPass(i);
            displayTrainResults(i);
      } 
      
      writeToFile();
      
   } //public void finishTraining()
   
   /**
    * Summarizes all the weights in a string so that the final print isn't as confusing
    * @return
    */
   public String showWeights()
   {
      return "Weights: " + Arrays.toString(weights[0][0]) + Arrays.toString(weights[0][1]) +
            Arrays.toString(weights[1][0]) + Arrays.toString(weights[1][1]);
   }

   
   /**
    * prints the results for each input pass
    * @param n in the index of the input being passed in 
    */
   public void displayRunResults(int n)
   {
      
      System.out.println("Run Complete - Inputs: " + (int)inputs[n][0] + " " +
               (int)inputs[n][1]);
      
      for (int i = 0; i < n_outputs; i++)
      {
         System.out.println("Output " + outNames[i] + ": " + outputs[i]);
      }
               
   } //public void displayRunResults(int n)
   
   
   /*
    * prints important results from training in a table like readable format
    */
   public void displayTrainResults(int n)
   {
         System.out.println("\nRun Complete - Inputs: " + (int)inputs[n][0] + " " +
               (int)inputs[n][1]);
         
         for (int i = 0; i < n_outputs; i++)
         {
            System.out.println("\t\tExpected Output :" + truthtable[n][i] +
               "\t\tOutput: " + outputs[i] + "\t\tError: " + errorVals[n][i]);
         }        

   } //public void displayTrainResults(int n)
   
   
   /*
    * prints the key details of the configuration of the network like
    * 
    * learning rate
    * dimensions
    * random ranges
    */
   public void displayNetworkConfig()
   {
      System.out.println("Lambda: " + lambda + "\nNumber of inputs: " + n_inputs + "\nNumber of hiddens :" + n_hiddens + 
            "\nNumber of outputs: " + n_outputs + "\nWeight generation information: Min value: " + 
            start + "\tMax value: " + end + "\nPreloaded weights: " + preload + "\nExecution time in ms: " + elapsed + " ms\n"); 
   }
   
   /*
    * writes all the information about the network and the new output weights
    * into a textfile in the same format as the input file
    */
   public void writeToFile()
   {
      try 
      {
         FileWriter fw = outputFile;
         
         //writing important info
         fw.write(n_inputs + "\n");
         fw.write(n_hiddens + "\n");
         fw.write(n_outputs + "\n");
         fw.write(lambda + "\n");
         fw.write(n_iters + "\n");
         fw.write(((train) ? 1 : 0) + "\n");
         fw.write(((preload) ? 1 : 0) + "\n");
         fw.write(start + "\n");
         fw.write(end + "\n");
         fw.write(e_thresh + "\n");
         
         fw.write("------------\n");
         
         //writing inputs
         
         fw.write(inputSetSize + "\n");
         
         for (int i = 0; i < inputSetSize; i++)
         {
            for (int j = 0; j < n_inputs; j++)
            {
               fw.write(inputs[i][j] + "\n");
               
            }
         } //for (int i = 0; i < inputSetSize; i++)
         
         //writing truthtable
         
         fw.write("------------\n");
         
         for (int i = 0; i < inputSetSize; i++)
         {
            for (int j = 0; j < n_outputs; j++)
            {
               fw.write(truthtable[i][j] + "\n");
            }
         } //for (int i = 0; i < inputSetSize; i++)
         
         //writing output weights
         
         fw.write("------------\n");
         
         for (int i = 0; i < n_inputs; i++)
         {
            for (int j = 0; j < n_hiddens; j++)
            {
               fw.write(weights[0][i][j] + "\n");
            }
         } //for (int i = 0; i < n_inputs; i++)
         
         for (int i = 0; i < n_hiddens; i++)
         {
            for (int j = 0; j < n_outputs; j++)
            {
               fw.write(weights[1][i][j] + "\n");     
            }
         } //for (int i = 0; i < n_hiddens; i++)
         
         fw.close();
         
      } //try 
      
      catch (IOException e) 
      {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
      
   } //public void writeToFile()
   
   /*
    * reads in everything that is needed from a specified text file
    */
   public void readfile()
   {  
      //checks that file is found
      Scanner sc = null;
      try 
      {
         sc = new Scanner(inputFile);
      } 
      catch (FileNotFoundException e) 
      {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
      
      //reading parameters
      
      int in = sc.nextInt();
      int hiddens = sc.nextInt();
      int outputs = sc.nextInt();
      double lam = sc.nextDouble();
      int numit = sc.nextInt();
      
      int training = sc.nextInt();
      int preloadWeights = sc.nextInt();
      double minr = sc.nextDouble();
      double maxr = sc.nextDouble();
      double thresh = sc.nextDouble();
      
      sc.nextLine();
      
      n_inputs = in;
      n_hiddens = hiddens;
      n_outputs = outputs;
      
      n_iters = numit;
      lambda = lam;
      
      start = minr;
      end = maxr;
      
      train = (training == 1);
      preload = (preloadWeights == 1);
      
      e_thresh = thresh;
      tot_thresh = n_outputs * 2 * e_thresh; //2 is a constant that I picked
      
      configureNetwork();
      
      sc.nextLine();
      
      //inputs
      
      inputSetSize = sc.nextInt();
      
      inputs = new double[inputSetSize][n_inputs];
      
      for (int i = 0; i < inputSetSize; i++)
      {
         for (int j = 0; j < n_inputs; j++)
         {
            inputs[i][j] = sc.nextDouble();
            
         }
      } //for (int i = 0; i < inputSetSize; i++)
      
      sc.nextLine();
      sc.nextLine();
      
      //truthtable   
      
      truthtable = new double[inputSetSize][n_outputs];
      errorVals = new double[inputSetSize][n_outputs];
      
      for (int i = 0; i < inputSetSize; i++)
      {
         for (int j = 0; j < n_outputs; j++)
         {
            truthtable[i][j] = sc.nextDouble();
         }
      } //for (int i = 0; i < inputSetSize; i++)
      
      sc.nextLine();
      sc.nextLine();
      
      //weights
      
      
      if (preload)
      {
         for (int i = 0; i < n_inputs; i++)
         {
            for (int j = 0; j < n_hiddens; j++)
            {
               weights[0][i][j] = sc.nextDouble();
            }
         } //for (int i = 0; i < n_inputs; i++)
         
         for (int i = 0; i < n_hiddens; i++)
         {
            for (int j = 0; j < n_outputs; j++)
            {
               weights[1][i][j] = sc.nextDouble();      
            }
         } //for (int i = 0; i < n_hiddens; i++)
      } //if (preload)
 
      sc.close();
      
   } //public void readfile()
   
   /**
    * Creates a network object and completes the full forward pass and
    * result display for all 4 inputs
    * 
    * Network object takes in parameters to make the network as customizable as possible
    *       description of corresponding parameters below
    * @param args the files to read and write from
    */
   public static void main(String[] args)
   {
      
      //input file
      File iFile = new File("C:\\Users\\arnav\\OneDrive\\XPS\\School Files\\12\\NNs\\control file stuff\\" + args[0]);
      
      
      //output file
      FileWriter oFile = null;
      try {
         oFile = new FileWriter("C:\\Users\\arnav\\OneDrive\\XPS\\School Files\\12\\NNs\\control file stuff\\" + args[1]);
      } catch (IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
      
      Network network = new Network(iFile, oFile);
      
      if (network.isTraining())
      {
         network.train();
      }
      else
      {
         network.run();
      }
      
   } //public static void main(String[] args)

} //public class Network
