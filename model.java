import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;


/** model: Contains methods for training, evaluating and predicting based on a previous model
 * */
public class model {

    /* xHatBuilder: does what it says on the tin, builds a matrix from a given input string, adding an x0 value of 1.0
     *
     *      Preconditions:
     *      - x is a string
     *
     *      Postconditions:
     *      - returns x as a matrix
     *
     * */
    private RealMatrix xHatBuilder(String xFileName, int N, int D){
        String x;
        String[] xSplit;
        D = D+1;
        RealMatrix outMatrix = new BlockRealMatrix(N, D);
        try {
            BufferedReader xReader = new BufferedReader(new FileReader(xFileName));
            for (int i = 0; i < N; i++) {

                x = xReader.readLine();
                if (x == null) break;
                // x0
                outMatrix.setEntry(i, 0, 1.0);
                for (int j = 1; j < D; j++) {
                    xSplit = x.split(" ");
                    outMatrix.setEntry(i, j, Double.parseDouble(xSplit[j - 1]));
                }

            }
            xReader.close();
        }catch (Exception e){
            System.out.println("Error: could not read from x's input file");
            return null;
        }
        return outMatrix;
    }


    /* matrixBuilder: creates a NxD matrix from the given input file.  Does not incorporate a y0
     *
     *      Preconditions:
     *      - none
     *
     *      Postconditions:
     *      - Returns an new, populated N x 1 matrix
     *      - If file could not be read, returns null instead
     *
     * */
    private RealMatrix matrixBuilder(String yFileName, int N, int D){
        String y;
        String[] ySplit;
        RealMatrix outMatrix = new BlockRealMatrix(N, D);
        try {
            BufferedReader yReader = new BufferedReader(new FileReader(yFileName));
            for (int i = 0; i < N; i++) {
                y = yReader.readLine();
                if(y == null) break;
                for(int j = 0; j < D; j++) {
                    ySplit = y.split(" ");
                    outMatrix.setEntry(i, j, Double.parseDouble(ySplit[j]));
                }
            }
            yReader.close();
        }catch (Exception e) {
            System.out.println("Error: could not read from the input file");
            return null;
        }
        return outMatrix;
    }

    /* polyFit: calculates and returns h(x) for higher order polynomials
     *
     *      Preconditions:
     *      - xHat and beta are initialized
     *      - xHat[0] == 1 (though with this formula, xHat[0]^0 == 1)
     *
     *      Postconditions:
     *      - y is returned (will be zero if x and beta are empty)
     *      - If len > K+1, then assumes that all subsequent x values are of the same power
     * */
    private RealMatrix polynomialFit(RealMatrix xHat, RealMatrix beta, int N, int D, int K) {
        int power;
        double y;
        RealMatrix outMatrix = new BlockRealMatrix(N, 1);
        for (int i = 0; i < N; i++){
            power = 0;
            y = 0.0;
            for(int j = 0; j < D+1; j++){
                y += Math.pow(xHat.getEntry(i, j), power) * beta.getEntry(0,j);
                if(power < K) power++;
            }
            outMatrix.setEntry(i, 0, y);
        }
        return outMatrix;
    }

    /* modelWriter: pretty print writer for printing out a given model
     *
     *      Preconditions:
     *      - outMatrix is initialized
     *      - outFileName is a string
     *      - delim is a valid string (less than ideal, but still useful if format changes)
     *
     *      Postconditions:
     *      - outMatrix is printed out to the specified destination file in scientific notation
     *      - Returns false if the file could not be opened, or if there is an issue writing the file, true otherwise
     *
     * */
    private boolean modelWriter(String outFileName, RealMatrix outMatrix, String delim){
        try {
            BufferedWriter outWriter = new BufferedWriter(new FileWriter(outFileName));
            NumberFormat formatter = new DecimalFormat("0.###E0");
            int D = outMatrix.getRowDimension();
            int i;
            for(i = 0; i < D-1; i++){
                outWriter.write(formatter.format(outMatrix.getEntry(i, 0)) + delim);
            }
            outWriter.write(formatter.format(outMatrix.getEntry(i, 0)) + "\n");

            outWriter.close();
            return true;
        } catch (Exception e){
            System.out.println("Error: could not open model file for writing");
            return false;
        }
    }

    /* evaluation: evaluates how accurately a trained model predicted data.
     *
     *      Preconditions:
     *      - args has at least 2 arguments, though less than 7 will result in failure
     *
     *      Postconditions:
     *      - Prints evaluation of the input dataset versus the raw data to the console, returning success
     *      - If any of the input files cannot be read, returns failure
     *
     * */
    public boolean evaluation(String[] args) {
        if (args.length == 7) {
            String xFileName = args[1];
            String yFileName = args[2];
            String inModelName = args[3];
            int N = Integer.parseInt(args[4]);
            int D = Integer.parseInt(args[5]);
            int K = Integer.parseInt(args[6]);
            NumberFormat formatter = new DecimalFormat("0.###E0");

            double total = 0.0;
            RealMatrix x = xHatBuilder(xFileName, N, D);
            RealMatrix y = matrixBuilder(yFileName, N, 1);
            RealMatrix beta = matrixBuilder(inModelName, 1, D+1);

            if(x == null || y == null || beta == null) return false;
            RealMatrix h = polynomialFit(x, beta, N, D, K);
            for(int i = 0; i < N; i++){
                total = total + Math.pow((h.getEntry(i, 0) - y.getEntry(i, 0)), 2);
            }
            total = total / N;
            System.out.println("Mean squared error of the model is: " + formatter.format(total));

            return true;
        } else {
            System.out.println("Error: invalid number of arguments (must be in the format: '-eval x.txt y.txt in.model N D K')");
            return false;
        }
    }

    /* prediction: generates a set of preconditions given x values and a trained model.
     *
     *      Preconditions:
     *      - args has 2 or more values, less than 7 will result in premature failure
     *
     *      Postconditions:
     *      - out.predictions is populated with a set of predictions from the data, returning success
     *      - Returns failure if any of the files cannot be opened
     *
     * */
    public boolean prediction(String[] args) {
        if (args.length == 7) {
            String xFileName = args[1];
            String inModelName = args[2];
            String outPredictionName = args[3];
            int N = Integer.parseInt(args[4]);
            int D = Integer.parseInt(args[5]);
            int K = Integer.parseInt(args[6]);

            RealMatrix x = xHatBuilder(xFileName, N, D);
            RealMatrix beta = matrixBuilder(inModelName, 1, D+1);
            RealMatrix y = polynomialFit(x, beta, N, D, K);

            return modelWriter(outPredictionName, y, "\n");
        } else{
            System.out.println("Error: invalid number of arguments (must be in the format: '-pred x.txt in.model out.predictions N D K')");
            return false;
        }
    }

    /* analyticalTrain: Analytical training, takes in input data and calculates beta matrix using formula:
     *                  beta = (xTx)^-1 * xTy
     *
     *      Preconditions:
     *      - input arguments are correct (ex: N, D, K are all ints, etc.)
     *
     *      Postconditions:
     *      - beta matrix is calculated and printed to an output file
     *      - All files used are closed
     *      - Returns false if any of the files cannot be opened, true otherwise
     * */
    private boolean analyticalTrain(String xFileName, String yFileName, String outFileName, int N, int D, int K) {
        RealMatrix xHat, xHatTranspose, xTx, xTy, y, beta;
        xHat = xHatBuilder(xFileName, N, D);
        y = matrixBuilder(yFileName, N, 1);
        if(xHat == null || y == null) return false;

        xHatTranspose = xHat.transpose();
        xTx = xHatTranspose.multiply(xHat);
        xTy = xHatTranspose.multiply(y);
        beta = MatrixUtils.inverse(xTx).multiply(xTy);

        return modelWriter(outFileName, beta, " ");

    }



    /*
     *
     * */
    private boolean checkConvergence(RealMatrix betaOld, RealMatrix betaNew, int N, int D, double st){
        if(betaNew == null) return false;
        double total = 0.0;
        for(int i = 0; i<N; i++){
            for(int j = 0; j < D; j++){
                total += (betaOld.getEntry(i,j) - betaNew.getEntry(i,j))/betaOld.getEntry(i,j);
            }
        }
        if(st > Math.abs(total)) return true;
        return false;
    }


    /* gradientTrain: gradient descent training using formula 2/N (xHatTranspose * xHat * betaHat) - 2/N (xHatTranspose * y)
     *                to calculate descent.
     *
     *      Preconditions:
     *      - Input variables are correct (ex: N, D, K are ints, etc)
     *
     *      Postconditions:
     *      - Best calculated beta printed out to a g
     *
     * */
    private boolean gradientTrain(String xFileName, String yFileName, String outFileName,
                                  double ss, double st, int N, int D, int K) {

        boolean converged = false;
        RealMatrix xHat, xHatTranspose, y, beta, newBeta, gradient, newGradient;
        xHat = xHatBuilder(xFileName, N, D);
        y = matrixBuilder(yFileName, N, 1);
        // initial betaHat full of zeroes.
        beta = new BlockRealMatrix(D+1, 1);
        if(xHat == null || y == null || beta == null) return false;

        xHatTranspose = xHat.transpose();
        gradient = xHatTranspose.multiply(y).scalarMultiply(-2.0/N)
                .add(xHatTranspose.multiply(xHat.multiply(beta)).scalarMultiply(2.0/N));
        while(!converged){
            newBeta = beta.subtract(gradient.scalarMultiply(ss));
            newGradient = xHatTranspose.multiply(y).scalarMultiply(-2.0/N)
                    .add(xHatTranspose.multiply(xHat.multiply(newBeta)).scalarMultiply(2.0/N));
            converged = checkConvergence(beta, newBeta, D+1, 1, st);
            beta = newBeta;
            gradient = newGradient;
        }


        return modelWriter(outFileName, beta, " ");
    }

    /* training: Method which deals in training the model.  Takes in a list of arguments, and assuming
     *           the input is properly formatted, generates a new model file with the specified name.
     *           Sends arguments off to the appropriate training helper method.
     *
     *      Preconditions:
     *      - args contains at least 2 arguments (needs 8 to proceed)
     *
     *      Postconditions:
     *      - out.model has been created and filled with a trained model, returning success
     *      - If out.model can't be written, or the input files cannot be read, then returns failure
     * */
    public boolean training(String[] args) {
        boolean success = false;
        //Setup for gradient descent
        if (args.length == 10 && args[4].equals("g")) {
            double ss = Double.parseDouble(args[5]);
            double st = Double.parseDouble(args[6]);
            success = gradientTrain(args[1], args[2], args[3], ss, st,
                    Integer.parseInt(args[7]), Integer.parseInt(args[8]), Integer.parseInt(args[9]));

        } else if (args.length == 8 && args[4].equals("a")) {
            success = analyticalTrain(args[1], args[2], args[3], Integer.parseInt(args[5]),
                    Integer.parseInt(args[6]), Integer.parseInt(args[7]));
        } else{
            System.out.println("Error: invalid number of arguments (must be in format: '-train x.txt y.txt out.model [a | g ss st] N D K");
        }
        return success;
    }
}
