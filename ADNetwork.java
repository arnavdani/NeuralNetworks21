import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;

/**
 * 
 * @author Arnav Dani
 * 
 * Created: 11-11-2021
 * 
 * Simulates the behavior of an A-B-C-D neural network that has one hidden layer
 * The hidden layer, input layer, and output layer can have any number of activations
 * 
 * The weights are optimized using a gradient descent and back propagation is utilized
 * to make the process as efficient as possible. 
 * 
 * Tanh is the new activation function, replacing sigmoid
 * 
 * In addition, file reading and writing is implemented.
 */

public class Network 
{
   double[] outputs;       //output activation
   double[][] inputs;      //input activations for xor
   double[] hiddens1;      //nodes in the 1st hidden layer
   double[] hiddens2;      //nodes in the 2nd hidden layer   
   double[][][] weights;   //weights in the network
   double[][] truthtable;  //truth table for xor
   String[] outNames;      //clarifies the output being evaluated (Or/and/xor etc)
   double[][] errorVals;   //store errors from each pass to use for error checks
   int n_inputs;           //constant with the number of sets of inputs, 4 for xor
   int n_hiddens1;         //constant for the number of hidden nodes in the first hidden layer
   int n_hiddens2;         //constant for nubmer of hidden nodes in the 2nd hidden layer         
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
   double[] thetak;  
   double[] psiLowerI;
   double[][] deltaWeightsJ;
   double e_thresh;        //error threshold for an individual case
   double tot_thresh;      //total combined error for all cases
   double omegaJ;
   double omegaK;
   double[] upperPsiJ;
   double upperPsiK;
   
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
    * reads input file and builds the network
    * 
    * @param inFile input file to read config from
    */
   public Network(File inFile)
   {     
      inputFile = inFile;
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
         thetaj = new double[n_hiddens2];
         thetai = new double[n_outputs];
         thetak = new double[n_hiddens1];
         psiLowerI = new double[n_outputs];
         omega = new double[n_outputs];
         upperPsiJ = new double[n_hiddens2];
      } //if (train)
      
      n_layers = 3;                          //excluding output layer since no connection leaves the output layer
      hiddens1 = new double[n_hiddens1];
      hiddens2 = new double[n_hiddens2];
      outputs = new double[n_outputs];
      
      dimCount = Math.max(n_inputs, n_hiddens1); 
      dimCount = Math.max(dimCount, n_hiddens2); 
      dimCount = Math.max(dimCount, n_outputs);           //to ensure all connections are represented
      
      weights = new double[n_layers][dimCount][dimCount]; //weights are between layers A-B B-C C-D 
      
      if (!preload)
      {
         randomizeWeights();
      }
         
      resetNetwork();
   } //public void configureNetwork()
   
   /**
    * identifies the type of targets
    * temporary method, only used for printing when training on binary gates
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
    * This describes the hyperbolic tangent function being used
    * All previously used functions are commented
    * 
    * @param x input into the function
    * @return type double of x passed through the function
    */
   public double activate(double x)
   {
      //sigmoid
      return  1.0 / (1.0 + Math.exp(-x));
      
      //hyperbolic tangent
      //double e2x = Math.exp(2 * x);
      //return (e2x - 1.0) / (e2x + 1.0);
   }
   
   /**
    * derivative of the activation function
    * 
    * the current activation function is hyperbolic tangent: the derivative is 1-f^2
    * 
    * @param x the input value to pass through the derivative
    * @return the value after being passed through the function 
    */
   public double actDeriv(double x)
   {
      //sigmoid
      double act = activate(x);
      return act * (1.0 - act);
      
      //hyperbolic tan
      //double act = activate(x);
      //return 1 - act * act;
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
      
      for (int k = 0; k < n_hiddens1; k++)
      {
         val = 0.0;    
         
         for (int m = 0; m < n_inputs; m++)
         {
            val += inputs[num][m] * weights[0][m][k];
         } //for (int m = 0; m < n_inputs; m++)
         
         hiddens1[k] = activate(val);
         
      } //for (int k = 0; k < n_hiddens1; k++)
      
      for (int j = 0; j < n_hiddens2; j++)
      {
         val = 0.0;
               
         for (int k = 0; k < n_hiddens1; k++)
         {
            val += hiddens1[k] * weights[1][k][j];
         } //for (int j = 0; j < n_hiddens2; j++)
         
         hiddens2[j] = activate(val);
      } //for (int j = 0; j < n_hiddens2; j++)
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
      for (int k = 0; k < n_hiddens1; k++)
      {
         val = 0.0;    
         
         for (int m = 0; m < n_inputs; m++)
         {
            val += inputs[num][m] * weights[0][m][k];
         } //for (int m = 0; m < n_inputs; m++)
         
         hiddens1[k] = activate(val);
         thetak[k] = val; 
      } //for (int k = 0; k < n_hiddens1; k++)
      
      for (int j = 0; j < n_hiddens2; j++)
      {
         val = 0.0;
               
         for (int k = 0; k < n_hiddens1; k++)
         {
            val += hiddens1[k] * weights[1][k][j];
         } //for (int k = 0; k < n_hiddens1; k++)
         
         thetaj[j] = val;
         hiddens2[j] = activate(val);
      } //for (int j = 0; j < n_hiddens2; j++)
   } //public void trainCalcHiddens(int num)
   
   /**
    * calculates the final output using the dot products of the weights
    * in the hidden layers
    * 
    * @param input identifies the binary input being used
    * 
    */
   public void calcOutput(int input)
   {
      
      for (int i = 0; i < n_outputs; i++)
      {
         double val = 0.0;
         
         for (int j = 0; j < n_hiddens2; j++)
         {
            val += hiddens2[j] * weights[2][j][i];
         } //for (int j = 0; j < n_hiddens2; j++)
         
         outputs[i] = activate(val);
         
      } //for (int i = 0; i < n_outputs; i++)
   } //public void calcOutput(int input)
   
   /**
    * used for training
    * 
    * calculates the final output and theta(i) values using the dot products of the weights
    * in the hidden layers
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
         
         for (int j = 0; j < n_hiddens2; j++)
         {
            val += hiddens2[j] * weights[2][j][i];
         } //for (int j = 0; j < n_hiddens2; j++)
         
         outputs[i] = activate(val);
         
         //training related values
         thetai[i] = val;
         omega[i] = getError(i, input);
         errorVals[input][i] = omega[i];
         psiLowerI[i] = omega[i] * actDeriv(thetai[i]);        
      } //for (int i = 0; i < n_outputs; i++)
   } //public void trainCalcOutput(int input)
   
   /**
    * Since there are 4 possible inputs for an Xor, n represents
    * which input is being tested to that the output of the network
    * can be correctly compared to the correct ground truth
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
      }//for (int i = 0; i < inputSetSize; i++)
      
   } //public void run()

   /*
    * training specific code
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
         } //for (int k = 0; k < dimCount; k++)
      } // for (int n = 0; n < n_layers; n++) 
   } //public void randomizeWeights()
   
   
   /**
    * Implements gradient descent and backpropgatation to calculate the optimal change for each
    * each and stores that amount to apply later
    * 
    * @param input identifies the input set - the integer helps identify whether the input was 
    *    0,0     0,1     1,0     1,1
    */
   public void calculateWeights(int input)
   {
      trainForwardPass(input);           //initial forward pass
      
      for (int j = 0; j < n_hiddens2; j++)
      {
         omegaJ = 0.0;    
         for (int i = 0; i < n_outputs; i++)
         {
            omegaJ += psiLowerI[i] * weights[2][j][i];            
            weights[2][j][i] += lambda * hiddens2[j] * psiLowerI[i];
         } //for (int i = 0; i < n_outputs; i++)
         
         upperPsiJ[j] = omegaJ * actDeriv(thetaj[j]);
         
      } //for (int j = 0; j < n_hiddens2; j++)
      
      for (int k = 0; k < n_hiddens1; k++)
      {
         omegaK = 0.0;
         for (int j = 0; j < n_hiddens2; j++)
         {
            omegaK += upperPsiJ[j] * weights[1][k][j];
            weights[1][k][j] += lambda * hiddens1[k] * upperPsiJ[j];      
         } //for (int j = 0; j < n_hiddens2; j++)
         
         upperPsiK = omegaK * actDeriv(thetak[k]);
         
         for (int m = 0; m < n_inputs; m++)
         {
            weights[0][m][k] += lambda * inputs[input][m] * upperPsiK; 
         }//for (int m = 0; m < n_inputs; m++)
         
      } //for (int k = 0; k < n_hiddens1; k++)
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
       * for all inputs - for example, if e_thresh is 0.05, the error must be <0.05 or <-0.05
       * 
       * Second - the sum of the absolute values of all errors should be less than #outputs  * threshold (tot_thresh)
       * 
       * Both are necessary to limit outliers
       * The thresholds are mutable via a parameter in control file
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
            maxError = Math.max(maxError, Math.abs(currentError));
            if (currentError > e_thresh || currentError < -e_thresh )
            {           
               errcheck = false;         
            }   
           
         } //for (int j = 0; j < inputSetSize; j++)
      } //for (int i = 0; i < n_outputs; i++)
      
      exitConditions[2] = (totalError < tot_thresh) && errcheck;
      
      /*
       * as more error checks are added, they will be processed here
       */
      
      /*
       * if there is one true error check, return true
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
    * This signifies that the loop should no longer be repeating
    */
   public void exit()
   {
      repeat = false;
   }
   
   /**
    * executes the full training sequence to train the network
    * to be able to recognize an XOR pattern
    */
   public void train()
   {     
      resetNetwork();
      setTargets();
      startTime = System.currentTimeMillis();
      
      while (repeat)
      {  
         int indexMod = curr_iters % inputSetSize;
         calculateWeights(indexMod);
         
         curr_iters++;
         
         if (indexMod == 3)   //only checks to exit after all 4 sets of inputs are passed
         {
            if (checkExit())
            {
               exit();
            }
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
      
      /*
       * add more prints for each added exit condition
       */
      
      System.out.println("\n\nNumber of iterations at ending: " + curr_iters);
      
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
         } //for (int i = 0; i < n_outputs; i++)        
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
      System.out.println("Lambda: " + lambda + "\nNumber of inputs: " + n_inputs + "\nNumber of hiddens1 :" + n_hiddens1 + 
            "\nNumber of hiddens2: " + n_hiddens2 + "\nNumber of outputs: " + n_outputs + "\nWeight generation information: Min value: " + 
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
         
         int one = 1;
         int zero = 0;
         
         //writing important info
         fw.write("output.txt\n");      //rewrites itself if retrained
         fw.write(n_inputs + "\n");
         fw.write(n_hiddens1 + "\n");
         fw.write(n_hiddens2 + "\n");
         fw.write(n_outputs + "\n");
         fw.write(lambda + "\n");
         fw.write(n_iters + "\n");
         fw.write(zero + "\n");       //there is no reason to retrain, so new file should not be trained with
         fw.write(one + "\n");        //weights must be fixed, not randomized, thats why training happened
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
         
         for (int m = 0; m < n_inputs; m++)
         {
            for (int k = 0; k < n_hiddens1; k++)
            {
               fw.write(weights[0][m][k] + "\n");          
            }
         } //for (int m = 0; m < n_inputs; m++)
         
         for (int k = 0; k < n_hiddens1; k++)
         {
            for (int j = 0; j < n_hiddens2; j++)
            {
               fw.write(weights[1][k][j] + "\n");     
            }
         } //for (int k = 0; k < n_hiddens1; k++)
         
         for (int j = 0; j < n_hiddens2; j++)
         {
            for (int i = 0; i < n_outputs; i++)
            {
               fw.write(weights[2][j][i] + "\n");     
            }
         } //for (int j = 0; j < n_hiddens2; j++)
         
         fw.close(); 
      } //try 
      
      catch (IOException e) 
      {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
   } //public void writeToFile()
   
   /*
    * reads in everything that is needed from a default or specified text file
    */
   public void readfile()
   {  
      //finds file
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
      String out = sc.nextLine();
      
    //output file
      try 
      {
         outputFile = new FileWriter("C:\\Users\\arnav\\OneDrive\\XPS\\School Files\\12\\NNs\\control file stuff\\" + out);
      } 
      catch (IOException e) 
      {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
          
      int in = sc.nextInt();
      int hiddens1 = sc.nextInt();
      int hiddens2 = sc.nextInt();
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
      n_hiddens1 = hiddens1;
      n_hiddens2 = hiddens2;
      n_outputs = outputs;
      n_iters = numit;
      lambda = lam;
      start = minr;
      end = maxr;
      
      train = (training == 1);
      preload = (preloadWeights == 1);
      
      e_thresh = thresh;
      tot_thresh = n_outputs * e_thresh; 
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
         for (int m = 0; m < n_inputs; m++)
         {
            for (int k = 0; k < n_hiddens1; k++)
            {
               weights[0][m][k] = sc.nextDouble();
            }
         } //for (int m = 0; m < n_inputs; m++)
         
         for (int k = 0; k < n_hiddens1; k++)
         {
            for (int j = 0; j < n_hiddens2; j++)
            {
               weights[1][k][j] = sc.nextDouble();
            }
         } //for (int k = 0; k < n_inputs; k++)
         
         for (int j = 0; j < n_hiddens2; j++)
         {
            for (int i = 0; i < n_outputs; i++)
            {
               weights[2][j][i] = sc.nextDouble();  
            }
         } //for (int j = 0; j < n_hiddens2; j++)
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
      File iFile = null;
      
      if (args.length > 0)        //checks if argument has been passed in
      {
      iFile = new File("C:\\Users\\arnav\\OneDrive\\XPS\\School Files\\12\\NNs\\control file stuff\\" + 
                        args[0]); //input file
      }
      else //default file
      {
         iFile = new File("C:\\Users\\arnav\\OneDrive\\XPS\\School Files\\12\\NNs\\control file stuff\\default_ABCD.txt");        
      }
      
      Network network = new Network(iFile);
      
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
