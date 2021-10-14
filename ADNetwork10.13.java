import java.util.Arrays;

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
 * The weights are optimized using a gradient descent
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
   final int N_INPUTS;     //constant with the number of sets of inputs, 4 for xor
   final int N_HIDDENS;    //constant for the number of hidden nodes
   final int N_LAYERS;     //number of layers
   final int N_OUTPUTS;    //number of outputs
   double start;           //start of random range
   double end;             //end of random range
   int inputSetSize;       //size of training set
   double maxError;        //largest error from a single pass
   double totalError;      //sum of all errors from a single pass
   
   
   
   /*
    * exit condition variables
    */
   
   final int N_ITERS;            //max number of iterations the network should pass through
   final int N_ERRORCHECKS = 3;
   int curr_iters;               //current number of iterations
   double lambda;                //learning rate
   boolean[] exitConditions;
   
   boolean repeat;               //says whether to do another training pass
   boolean train;                //says whether the network is training or running
   
   
   /*
    * training variables
    */
   double[] omega;
   double[] thetai;
   double[] thetaj;        
   double[] psiLowerI;
   double partialsJ;
   double deltaWeightJ;
   double[][] deltaWeightsJ;
   final double E_THRESH;        //error threshold for an individual case
   final double TOT_THRESH;      //total combined error for all cases
   
   double omegaJ;
   double upperPsiJ;
   double partialsKJ;
   double deltaWeightKJ;
   double[][] deltaWeightsK;
   
      
   /**
    * Constructor for the network class
    * calls the method that sets up all the fundamental arrays 
    * and initializes the output variable
    * 
    * @param training determines whether the network is training or running
    * @param numIters the max number of iterations for the network
    * @param initialLambda the initial learning rate
    * @param randstart double for min value for random gen
    * @param randend double max value for random gens
    * @param numin number of inputs
    * @param numhid number of hidden nodes
    */
   public Network(boolean training, int numIters, double initialLambda, double randstart, double randend, int numin, int numhid, int numout, double tresh)
   {
      train = training;
      N_HIDDENS = numhid;
      N_INPUTS = numin;
      N_OUTPUTS = numout;
      N_LAYERS = 2;           //excluding output layer since no connection leaves the output layer
      
      
      
      E_THRESH = tresh;
      TOT_THRESH = N_OUTPUTS * 2 * E_THRESH; //2 is a constant that I picked
      
      
      configureNetwork();
         
      N_ITERS = numIters;
      curr_iters = 0;
      lambda = initialLambda;
      repeat = true;
      
      start = randstart;
      end = randend;
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
    */
   public void configureNetwork()
   {  
      hiddens = new double[N_HIDDENS];
      thetaj = new double[N_HIDDENS];
      thetai = new double[N_OUTPUTS];
      deltaWeightsJ = new double[N_HIDDENS][N_OUTPUTS];
      deltaWeightsK = new double[N_INPUTS][N_HIDDENS];
      psiLowerI = new double[N_OUTPUTS];
      outputs = new double[N_OUTPUTS];
      omega = new double[N_OUTPUTS];
      
      int dimCount = Math.max(N_INPUTS, N_HIDDENS); //to ensure all connections are represented
      
      weights = new double[N_LAYERS][dimCount][dimCount]; //weights are between layers A-B and B-C
      
      setWeights(); 
      
      resetNetwork();
   }
   
   
   /**
    * Sets the individual values of each weight
    * this is where the weights can be hard coded to certain values
    * 
    * This hard coding only works for a 2-5-3 architecture
    * 
    * 
    */
   public void setWeights()
   {
      /*
       * the values of each weight can/should be changed
       */
      double w000 = 0.2;
      double w001 = 0.3;
      double w002 = 0.4;
      double w003 = 0.4;
      double w004 = 0.4;
      
      double w010 = 0.5;
      double w011 = 0.7;
      double w012 = 0.4;
      double w013 = 0.4;
      double w014 = 0.4;
      
      double w100 = 0.4;
      double w101 = 0.2;
      double w102 = 0.5; 
      
      double w110 = 0.4;
      double w111 = 0.2;
      double w112 = 0.5;    
      
      double w120 = 0.4;
      double w121 = 0.2;
      double w122 = 0.5;    
      
      double w130 = 0.4;
      double w131 = 0.2;
      double w132 = 0.5;    
      
      double w140 = 0.4;
      double w141 = 0.2;
      double w142 = 0.5;    
      
      /*
       * hard coding the weights
       */
      weights[0][0][0] = w000; 
      weights[0][0][1] = w001; 
      weights[0][0][2] = w002; 
      weights[0][0][3] = w003; 
      weights[0][0][4] = w004; 
      
      weights[0][1][0] = w010;
      weights[0][1][1] = w011;
      weights[0][1][2] = w012;
      weights[0][1][3] = w013;
      weights[0][1][4] = w014;
      

      weights[1][0][0] = w100;
      weights[1][0][1] = w101;
      weights[1][0][2] = w102;
      
      
      weights[1][1][0] = w110;
      weights[1][1][1] = w111;
      weights[1][1][2] = w112;
      
      weights[1][2][0] = w120;
      weights[1][2][1] = w121;
      weights[1][2][2] = w122;
      
      weights[1][3][0] = w130;
      weights[1][3][1] = w131;
      weights[1][3][2] = w132;
      
      weights[1][4][0] = w140;
      weights[1][4][1] = w141;
      weights[1][4][2] = w142;
   }
   
   /**
    * initializes the arrays of inputs
    * an XOR has 4 possible combinations of inputs
    * 
    * 
    * In total, row 0 represents (0, 0)
    * row 1 represents (0, 1)
    * row 2 represents (1, 0)
    * row 3 represents (1, 1)
    * 
    * This only works when there are 4 sets of inputs with size 2
    * 
    * This is where the inputs are hard coded in
    * 
    */
   public void setInputs()
   {
      
      inputSetSize = 4; //input size is 4 because that is the number of combinations
      inputs = new double[inputSetSize][N_INPUTS];
                     
      /**
       * for error calculation - 4 forward passes so 4 actual true outputs
       * are needed to compare to the outputs of the pass
       */
                  
      inputs[0][0] = 0.0;
      inputs[0][1] = 0.0;
      
      inputs[1][0] = 0.0;
      inputs[1][1] = 1.0;
      
      inputs[2][0] = 1.0;
      inputs[2][1] = 0.0;
      
      inputs[3][0] = 1.0;
      inputs[3][1] = 1.0;        
      
   }
   
   /**
    * Initializes the array of ground truths following the same convention
    * of inputs as setInputs()
    * 
    * Since each of the 3 outputs need to be trained to a seperate boolean problem,
    * truthtable is a 2d array to represent the unique truth table for each output
    * 
    * 
    * I am going in the order of OR, AND, XOR
    * 
    * This is where the truthtable is hardcoded
    */
   public void setTargets()
   {
      truthtable = new double[N_OUTPUTS][inputSetSize];
      errorVals = new double[N_OUTPUTS][inputSetSize];
      
      
      /*
       * outNames is used for prints and clarification
       */
      outNames = new String[N_OUTPUTS];
      outNames[0] = "OR";
      outNames[1] = "AND";
      outNames[2] = "XOR";
      
      /*
       * OR based on the inputs
       */
      truthtable[0][0] = 0;
      truthtable[0][1] = 1;
      truthtable[0][2] = 1;
      truthtable[0][3] = 1;
      
      /*
       * AND based on the inputs
       */
      truthtable[1][0] = 0;
      truthtable[1][1] = 0;
      truthtable[1][2] = 0;
      truthtable[1][3] = 1;
      
      
      /*
       * XOR based on the inputs
       */
      truthtable[2][0] = 0;
      truthtable[2][1] = 1;
      truthtable[2][2] = 1;
      truthtable[2][3] = 0;
   }
      
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
   }
   
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
    * calculates the value of the activations in the hidden layers
    * using dot products of the weights of the input layers
    * 
    * uses variable val to avoid += from messing up values in the hidden layer
    * @param num refers to the input type
    */
   public void calcHiddens(int num)
   {
      double val = 0.0;
      for (int j = 0; j < N_HIDDENS; j++)
      {
         val = 0.0;    
         
         for (int k = 0; k < N_INPUTS; k++)
         {
            val += inputs[num][k] * weights[0][k][j];
         }
         
         hiddens[j] = activate(val);
         thetaj[j] = val;
         
      } //for (int j = 0; j < N_HIDDENS; j++)
   }
   
   /**
    * calculates the final output and theta(i) values using the dot products of the weights
    * in the hidden layer
    * 
    */
   public void calcOutput()
   {
      
      for (int i = 0; i < N_OUTPUTS; i++)
      {
         double val = 0.0;
         
         for (int j = 0; j < N_HIDDENS; j++)
         {
            val += hiddens[j] * weights[1][j][i];
         }
         
         outputs[i] = activate(val);
         thetai[i] = val;
         
      } //for (int i = 0; i < N_OUTPUTS; i++)
   }
   
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
      double error = truthtable[out][num] - outputs[out];
      return error;
   }
   
   /**
    * combines the methods defined to perform a forward pass through the network
    * @param the input being considered [(0, 1), (1, 1), etc]
    */
   public void forwardPass(int num)
   {
      calcHiddens(num);
      calcOutput();
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
      setInputs();
      setTargets(); //temporary - helps split between the 3 output cases
      
      for (int i = 0; i < N_OUTPUTS; i++)
      {
         System.out.println("\n\n" + outNames[i]+ "\n");
         for (int j = 0; j < inputSetSize; j++)
         {
            forwardPass(j);
            displayRunResults(i, j);
         }
      } //for (int i = 0; i < N_OUTPUTS; i++)
   }

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
      for (int n = 0; n < N_LAYERS; n++)
      {
         for (int k = 0; k < N_INPUTS; k++)
         {
            for (int j = 0; j < N_HIDDENS; j++)
            {
               weights[n][k][j] = randomgen(start, end); 
            }
         } //for (int k = 0; k < N_INPUTS; k++)
      } // for (int n = 0; n < N_LAYERS; n++)
      
   }
   
   /**
    * loads in the input and target data used to train the neural network
    * 
    * right now, this is the same as is for run, but loads target as a procedural to make code more
    *    understandable and readable
    */
   public void loadTrainingData()
   {
      setInputs();
      setTargets();        
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
      forwardPass(input);           //have to do an initial forward pass to calculate error and access correct activations
      
      
      /*
       * calculating the change in  the weights connecting the 
       * hidden layer to the output layer
       */
      
      for (int i = 0; i < N_OUTPUTS; i++)
      {
         omega[i] = getError(i, input);
         errorVals[i][input] = omega[i];
         
         psiLowerI[i] = omega[i] * actDeriv(thetai[i]); 
         
         
         for (int j = 0; j < N_HIDDENS; j++)
         {
            partialsJ = -hiddens[j] * psiLowerI[i];
            deltaWeightJ = -lambda * partialsJ;
            deltaWeightsJ[j][i] = deltaWeightJ;
            
         } 
      } //for (int i = 0; i < N_OUTPUTS; i++)
      
      
      /*
       * calculating the change in the weights connecting the 
       * input layer to the hidden layer
       */
      
      for (int j = 0; j < N_HIDDENS; j++)
      {
         omegaJ = 0.0;    
         for (int i = 0; i < N_OUTPUTS; i++)
         {
            omegaJ += psiLowerI[i] * weights[1][j][i];            
         }
         
         upperPsiJ = omegaJ * actDeriv(thetaj[j]);
         
         for (int k = 0; k < N_INPUTS; k++)
         {
            partialsKJ = -inputs[input][k] * upperPsiJ;           
            deltaWeightKJ = -lambda * partialsKJ;           
            deltaWeightsK[k][j] = deltaWeightKJ;   
         }
         
      } //for (int j = 0; j < N_HIDDENS; j++)
   }
   
   
   /**
    * minimizes the weights using the calculations from calculateWeights
    */
   public void minimizeWeights()
   {  
      /*
       * hiddens to output
       */
      
      for (int i = 0; i < N_OUTPUTS; i++)
      {
         for (int j = 0; j < N_HIDDENS; j++)
         {
            weights[1][j][i] += deltaWeightsJ[j][i];
         }
      } //for (int i = 0; i < N_OUTPUTS; i++)
      
      
      
      /*
       * input to hiddens
       */
      for (int k = 0; k < N_INPUTS; k++)
      {
         for (int j = 0; j < N_HIDDENS; j++)
         {
            weights[0][k][j] += deltaWeightsK[k][j];
         }
      } //for (int k = 0; k < N_INPUTS; k++)
   }
   
   /**
    * temporary method to "save" the weights
    * 
    * Currently, it prints the weights out
    * 
    * Eventually, the weights should be written and saved to a file
    */
   public void saveWeights()
   {
      System.out.println(showWeights());     
   }
      
   
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
      if (curr_iters >= N_ITERS)
            exitConditions[1] = true;     
      
      /*
       * check 3 - error thresholding
       * 
       * The error check I use has 2 steps
       * 
       * First - every error value must be less than the absolute value of an error threshold
       * for all 4 inputs - for example, if E_THRESH is 0.05, the error must be <0.05 or <-0.05
       * 
       * Second - the sum of the absolute values of all errors should be less than 2 times the threshold (TOT_THRESH)
       * 
       * Both are necessary - only checking total error could allow outliers to pass through
       * and only checking individual error requires a higher threshold to precision
       * 
       * The thresholds are mutable via a parameter passed through the constructor
       */
      double totalerr = 0.0;
      double currentError = 0.0;
      boolean errcheck = true;
      
      for (int i = 0; i < N_OUTPUTS; i++)
      {
         for (int j = 0; j < inputSetSize; j++)
         {
            currentError = errorVals[i][j];
            totalerr += Math.abs(currentError);
            if (currentError > E_THRESH || currentError < -E_THRESH )
            {           
               errcheck = false;
            }   
            else
            {
               maxError = Math.max(maxError, Math.abs(currentError));
            }
           
         } //for (int j = 0; j < inputSetSize; j++)
      } //for (int i = 0; i < N_OUTPUTS; i++)
      
      
      if (totalerr < TOT_THRESH || totalerr < -TOT_THRESH)
      {
         exitConditions[2] = errcheck;
         totalError = totalerr;
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
   }
   
   /**
    * exits the training loop by changing the repeat boolean to false
    * 
    * This signifies that the loop should no longer be repeating
    * 
    * the method exists to make the code more readable
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
      randomizeWeights();  
      loadTrainingData();
      
      while (repeat)
      {  
         calculateWeights(curr_iters % inputSetSize);
         minimizeWeights();
         
         curr_iters++;
         
         if (checkExit())
         {
            exit();
         }
            
      } //while (repeat)
      
      finishTraining();
      
   }
   
   /*
    * getting results and printing
    * 
    * Lists all the reasons why the training sequence terminated and 
    * then prints information about the network and results
    */
   
   public void finishTraining()
   {  

      displayNetworkConfig();
      
      System.out.println("Training done\nReason for finishing:\n");
      
      if (exitConditions[0] == true)
         System.out.println("Lambda became 0");
      
      
      if (exitConditions[1] == true)
         System.out.println("Max number of iterations of " + N_ITERS + " approached");
      
      if (exitConditions[2] == true)
         System.out.println("Error Threshold met - the highest magnitude of error for an individual case was " +
                     maxError + " which is less than " + 
                     E_THRESH + "\n\t\tCombined error was " + totalError + " which is less than " + TOT_THRESH +
                     "\n\nBoth of these conditions had to be met for termination");
      
      
      System.out.println("\n\nNumber of iterations at ending: " + curr_iters);
      
      /*
       * add more prints for each added exit condition
       */
      
      
      /*
       * last forward pass for results
       */
      
      for (int i = 0; i < N_OUTPUTS; i++)
      {
         System.out.println("\n\n" + outNames[i] + "\n");
         for (int j = 0; j < 4; j++)
         {
            forwardPass(j);
            displayTrainResults(i, j);
         }
         
      } //for (int i = 0; i < N_OUTPUTS; i++)
      
      
      //saveWeights(); - will be uncommented when file writing is done
   }
   
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
   public void displayRunResults(int out, int n)
   {
      
      System.out.println("Run Complete - Inputs: " + (int)inputs[n][0] + " " +
               (int)inputs[n][1] +  
               "\n\tOutput: " + outputs[out]);
   }
   
   /*
    * prints important results from training in a table like readable format
    */
   public void displayTrainResults(int out, int n)
   {
         System.out.println("Run Complete - Inputs: " + (int)inputs[n][0] + " " +
               (int)inputs[n][1] +  
               "\t\tExpected Output :" + truthtable[out][n] +
               "\t\tOutput: " + outputs[out] + "\t\tError: " + getError(out, n));

   }
   
   /*
    * prints the key details of the configuration of the network like
    * 
    * learning rate
    * dimensions
    * random ranges
    */
   public void displayNetworkConfig()
   {
      System.out.println("Lambda: " + lambda + "\nNumber of inputs: " + N_INPUTS + "\nNumber of hiddens :" + N_HIDDENS + 
            "\nNumber of outputs: " + N_OUTPUTS + "\n\nWeight generation information \n\tMin value: " + 
            start + "\tMax value: " + end + "\n\n"); 
   }
   
   /**
    * Creates a network object and completes the full forward pass and
    * result display for all 4 inputs
    * 
    * Network object takes in parameters to make the network as customizable as possible
    *       description of corresponding parameters below
    * @param args
    */
   public static void main(String[] args)
   {
      /*
       * parameters in order: 
       * 
       * training, num iters, lamba, min rand range, max rand range, # in input layer, 
       *          # in hidden layer, # in output layer, error threshold
       * 
       * 
       */
      
      int numIt = 200000;
      double lam = 0.3;
      double minR = -1.0;
      double maxR = 1.5;
      int inNum = 2;
      int hidNum = 5;
      int outNum = 3;
      double eThresh = 0.1;
      
      
      /*
       * to run network, set first parameter to false
       */
      Network network = new Network(true, numIt, lam, minR, maxR, inNum, 
               hidNum, outNum, eThresh);
      
      if (network.isTraining())
         network.train();
      else
         network.run();
      
   }

}
