import java.util.Arrays;


/**
 * 
 * @author Arnav Dani
 * 
 * Created: 9-7-2021
 * 
 * Simulates the behavior of a neural network that has one hidden layer and a single scalar output
 * The hidden layer and input layer can have any number of activations
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
	double F0; 					//output activation
	double[][] inputs; 		//input activations for xor
	double[] hiddens; 		//nodes in the 1 hidden layer
	double[][][] weights; 	//weights in the network
	double[] truthtable;		//truth table for xor
	double[] errorVals;		//store errors from each pass to use for error checks
	final int N_INPUTS;		//constant with the number of sets of inputs, 4 for xor
	final int N_HIDDENS;		//constant for the number of hidden nodes
	final int N_LAYERS;
	double start;
	double end;
	int inputSetSize;
	
	
	
	/*
	 * exit condition variables
	 */
	
	final int N_ITERS;				//max number of iterations the network should pass through
	final int N_ERRORCHECKS = 3;
	int curr_iters;					//current number of iterations
	double lambda;						//learning rate
	boolean[] exitConditions;
	
	boolean repeat;					//says whether to do another training pass
	
	
	/*
	 * training variables
	 */
	double omega0;
	double theta0;
	double[] thetaj;			
	double psiLower0;
	double partialsJ0;
	double deltaWeightJ0;
	double[] deltaWeightsJ;
	final double E_THRESH;
	final double TOT_THRESH;
	
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
	 * @param numIters the max number of iterations for the network
	 * @param initialLambda the initial learning rate
	 * @param randstart double for min value for random gen
	 * @param randend double max value for random gens
	 * @param numin number of inputs
	 * @param numhid number of hidden nodes
	 */
	public Network(int numIters, double initialLambda, double randstart, double randend, int numin, int numhid, double tresh)
	{
		
		N_HIDDENS = numhid;
		N_INPUTS = numin;
		N_LAYERS = 2;			//excluding output
		E_THRESH = tresh;
		TOT_THRESH = 4 * E_THRESH;
		configureNetwork(N_HIDDENS);
			
		N_ITERS = numIters;
		curr_iters = 0;
		
		lambda = initialLambda;
		repeat = true;
		
		start = randstart;
		end = randend;
			
		F0 = 0.0;	//zeroing the output; just in case
	}
	
	/**
	 * Initializes all the arrays, sets the hard coded weights, and updates all the error checks
	 * 
	 * @param numHiddens the number of hidden layers
	 */
	public void configureNetwork(int numHiddens)
	{	
		hiddens = new double[numHiddens];
		thetaj = new double[numHiddens];
		deltaWeightsJ = new double[numHiddens];
		deltaWeightsK = new double[N_INPUTS][numHiddens];
		
		int dimCount = Math.max(N_INPUTS, N_HIDDENS);
		
		weights = new double[N_LAYERS][dimCount][dimCount];
		
		setWeights();
		
		/*
		 * initializing all the exit conditions to false
		 */
		resetNetwork();
		
		
	}
	
	
	/**
	 * Sets the individual values of each weight
	 * this is where the weights can be hard coded to certain values
	 * 
	 * This hard coding only works for a 2-2-1 architecture
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
		double w010 = 0.5;
		double w011 = 0.7;
		
		double w100 = 0.2;
		double w110 = 0.5;
		
		/*
		 * hard coding the weights for forward pass
		 */
		weights[0][0][0] = w000; 
		weights[0][0][1] = w001; 
		weights[0][1][0] = w010;
		weights[0][1][1] = w011;

		weights[1][0][0] = w100;
		weights[1][1][0] = w110;		
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
		
		inputSetSize = 4;
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
	 * only works when there is a scalar output
	 * 
	 * This is where the truthtable is hardcoded
	 */
	public void setTargets()
	{
		truthtable = new double[inputSetSize];
		errorVals = new double[inputSetSize];
		
		/*
		 * XOR based on the inputs
		 */
		truthtable[0] = 0;
		truthtable[1] = 1;
		truthtable[2] = 1;
		truthtable[3] = 0;
	}
		
	/**
	 * resets all mutable instance variables to their initial, starting value
	 * 
	 * Currently, this does minimal work but as more variables get added, there
	 * 		will be more to reset
	 */
	public void resetNetwork()
	{
		exitConditions = new boolean[4];
		curr_iters = 0;
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
	 * @param n
	 */
	public void calcHiddens(int num)
	{
		double val;
		for (int j = 0; j < N_HIDDENS; j++)
		{
			val = 0;		
			for (int k = 0; k < N_INPUTS; k++)
			{
				val += inputs[num][k] * weights[0][k][j];
			}
			hiddens[j] = activate(val);
			thetaj[j] = val;
		}
	}
	
	/**
	 * calculates the final output using the dot products of the weights
	 * in the hidden layer
	 * 
	 * @return double output without activation function which is used in the training sequence
	 */
	public void calcOutput()
	{
		
		double val = 0;
		for (int j = 0; j < N_HIDDENS; j++)
		{
			val += hiddens[j] * weights[1][j][0];
		}
		
		F0 = activate(val);
		theta0 = val;
	}
	
	/**
	 * 
	 * Since there are 4 possible inputs for an Xor, n represents
	 * which input is being tested to that the output of the network
	 * can be correctly compared to the correct ground truth
	 * 
	 * 
	 * @param n the row in the truth table being evaluated
	 * @return double representing the error after passed through the function
	 */
	public double getError(int num)
	{
		double error =  truthtable[num] - F0;
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
		
		for (int i = 0; i < inputSetSize; i++)
		{
			forwardPass(i);
			displayRunResults(i);
		}
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
			}
		}
		
	}
	
	/**
	 * loads in the input and target data used to train the neural network
	 * 
	 * right now, this is the same as is for run, but loads target as a procedural to make code more
	 * 	understandable and readable
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
	 * 	0,0     0,1     1,0     1,1
	 * 
	 */
	public void calculateWeights(int input)
	{
		forwardPass(input);				//have to do an initial forward pass to calculate error and access activations
		
		
		
		/*
		 * calculating the change in  the weights connecting the 
		 * hidden layer to the output layer
		 */
		
		omega0 = getError(input);
		errorVals[input] = omega0;
		
		psiLower0 = omega0 * actDeriv(theta0); 	//theta0 was calculated along with the output in calcOutput, stored as class variable
			
		for (int j = 0; j < N_HIDDENS; j++)
		{
			partialsJ0 = -hiddens[j] * psiLower0;
			deltaWeightJ0 = -lambda * partialsJ0;
			deltaWeightsJ[j] = deltaWeightJ0;
			
		}
	
		
		
		/*
		 * calculating the change in the weights connecting the 
		 * input layer to the hidden layer
		 */
		
		for (int j = 0; j < N_HIDDENS; j++)
		{
			omegaJ = psiLower0 * weights[1][j][0];		
			upperPsiJ = omegaJ * actDeriv(thetaj[j]);
			
			for (int k = 0; k < N_INPUTS; k++)
			{
				partialsKJ = -inputs[input][k] * upperPsiJ;				
				deltaWeightKJ = -lambda * partialsKJ;				
				deltaWeightsK[k][j] = deltaWeightKJ;	
			}	
		}
	}
	
	
	/**
	 * minimizes the weights using the calculations from calculateWeights
	 */
	public void minimizeWeights()
	{	
		/*
		 * hiddens to output
		 */
		for (int j = 0; j < N_HIDDENS; j++)
		{
			weights[1][j][0] += deltaWeightsJ[j];
		}
		
		
		/*
		 * input to hiddens
		 */
		for (int k = 0; k < N_INPUTS; k++)
		{
			for (int j = 0; j < N_HIDDENS; j++)
			{
				weights[0][k][j] += deltaWeightsK[k][j];
			}
		}
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
		 * Second - the absolute value sum of all 4 errors should be less than 2 times the threshold (TOT_THRESH)
		 * 
		 * Both are necessary - the second one by itself could have large errors that are positive and negative
		 * but sum to small values - the first one by itself would allow for more room for error
		 * 
		 * The threshold are mutable via a parameter passed through the constructor
		 */
		double totalerr = 0;
		boolean errcheck = true;
		for (int i = 0; i < inputSetSize; i++)
		{
			totalerr += errorVals[i];
			if (errorVals[i] > E_THRESH || errorVals[i] < -E_THRESH )
			{				
				errcheck = false;
			}
		}
		if (totalerr < TOT_THRESH || totalerr < -TOT_THRESH)
			exitConditions[2] = errcheck;
		
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
			// % 4 because there are 4 sets of inputs
			calculateWeights(curr_iters % inputSetSize);
			minimizeWeights();
			
		
			curr_iters++;
			
			if (checkExit())
			{
				exit();
			}
				
		}
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
		
		System.out.println("Training done\nReason for finishing\n");
		
		if (exitConditions[0] == true)
			System.out.println("Lambda became 0");
		
		
		if (exitConditions[1] == true)
			System.out.println("Max number of iterations of " + N_ITERS + " approached");
		
		if (exitConditions[2] == true)
			System.out.println("Error Threshold met - all sets of inputs had an error less than " + 
							E_THRESH + " and a combined error less than " + TOT_THRESH);
		
		
		System.out.println("\nNumber of iterations at ending: " + curr_iters);
		
		/*
		 * add more prints for each added exit condition
		 */
		
		
		/*
		 * last forward pass for results
		 */
		for (int i = 0; i < 4; i++)
		{
			forwardPass(i);
			displayTrainResults(i);
		}
		
		
		/*
		 * currently a performative print
		 * 
		 * save weights should eventually save the weights to a file
		 */
		saveWeights();
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
	public void displayRunResults(int n)
	{
		System.out.println("Run Complete - Inputs: " + (int)inputs[n][0] + " " +
					(int)inputs[n][1] +	
					"\nInfo: \n\tweights: " + 
					showWeights() + 
					"\n\tOutput: " + F0);
	}
	
	/*
	 * prints important results from training in a table like readable format
	 */
	public void displayTrainResults(int n)
	{
		System.out.println("\n\nRun Complete - Inputs: " + (int)inputs[n][0] + " " +
					(int)inputs[n][1] +	
					"\t\tExpected Output :" + truthtable[n] +
					"\t\tOutput: " + F0 + "\t\tError: " + getError(n));
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
				"\nNumber of outputs: 1" + "\n\nWeight generation information \n\tMin value: " + start + "\tMax value: " + end + "\n\n"); 
	}
	
	/**
	 * Creates a network object and completes the full forward pass and
	 * result display for all 4 inputs
	 * 
	 * Network object takes in parameters to make the network as customizable as possible
	 * 		description of corresponding parameters below
	 * @param args
	 */
	public static void main(String[] args)
	{
		/*
		 * parameters in order: 
		 * 
		 * num iters, lamba, min rand range, max rand range, # in input layer, # in hidden layer, error threshold
		 */
		Network network = new Network(200000, 0.1, 0.1, 1.5, 2, 5, 0.1);
		network.train();
		
	}

}
