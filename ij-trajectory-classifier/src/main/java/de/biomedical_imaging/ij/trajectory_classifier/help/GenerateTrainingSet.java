package de.biomedical_imaging.ij.trajectory_classifier.help;

import de.biomedical_imaging.traJ.Trajectory;
import de.biomedical_imaging.traJ.features.BoundednessFeature;
import de.biomedical_imaging.traJ.simulation.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class GenerateTrainingSet {
    /*
     *  This script generates training trajectories
     *
     *  There are 4 different diffusion modes:
     *   - Free diffusion
     *   - Subdiffusion
     *   - Confined diffusion
     *   - Directed motion
     *
     *  For each trajectory:
     *  The signal to noise ratio is randomly chosen between 1 and 20
     *  For confined diffusion trajectories, the boundedness is chosen randomly between 1 and 8. W
         with the given boundedness value a confinement radius is derived.
     *  For subdiffusion trajectories, alpha is chosen randomly between 0.7 and 0.3
     *  For directed motion, the ratio between active transport and diffusion is randomly chosen between 1 and 18.
     *
     */
    enum SIM_TYPE {
        FREE_rotation,
        CONFINED1_TA,
        CONFINED2_TR,
        ACTIVE_DMR,
        ACTIVE_DM
    }

    private static CentralRandomNumberGenerator r;

    public static void main(String[] args) {

        final int MODE_TRAINING = 1;
        final int MODE_TEST = 2;
        final int MODE_VALIDATION = 3;

        // Set the desired mode here
        int mode = MODE_TEST;  // can be switched dynamically

        //General Parameters
        int numberOfTracks = 0;
        int seed = 0;
        String prefix = "";

        switch (mode) {
            case MODE_TRAINING:
                numberOfTracks = 10000;     // num traces of each type
                seed = 22;
                prefix = "training";
                break;
            case MODE_TEST:
                numberOfTracks = 1250;
                prefix = "test";
                seed = 23;
                break;
            case MODE_VALIDATION:
                numberOfTracks = 1250;
                prefix = "validation";
                seed = 24;
                break;

            default:
                break;
        }
        String path = "C:\\Users\\songn\\Desktop\\Traj\\CLS\\2D\\SNR03\\" + prefix + ".txt";
        r = CentralRandomNumberGenerator.getInstance();
//        r.setSeed(seed);   // Set the seed to ensure the generated random numbers are reproducible.
        int num_class = 3;
        int dimension = 2;
        double timelag = 0.02; //s  Time resolution
        double D_ND = 0.22; //[µm^2/s];   Normal diffusion
        double D_TA = 0.0017; // Tight attachment
        double D_TR = 0.004; // Tether rotation
        double D_DM = 0.08; // Active transport (Directed Motion)
        double D_DMR = 0.05; //  Active transport (Directed Motion) and rotation
        double angleVelocity = Math.PI / 12.0; // rad/s   For Active transport

        SIM_TYPE[] types = SIM_TYPE.values();
        ArrayList<Trajectory> trajectorys = new ArrayList<Trajectory>();

        System.out.println("Generation of simulation trajectories");
        int tCounter = 0;
        for (SIM_TYPE type : types) {
            for (int i = 0; i < numberOfTracks; i++) {
                double tracklength = (1 + r.nextDouble() * 5);
//                double tracklength = 4;          // For each timelag, so 200 = 4 * 50ms;
                int numberOfSteps = (int) (tracklength * 1 / timelag);
                double boundedness = 1 + r.nextDouble() * 5;
                double alpha = 0.3 + r.nextDouble() * 0.4;    // 0.3 - 0.7
                AbstractSimulator sim = null;
                String typestring = "";
                typestring += type.toString();
                Trajectory t = null;
//                double diffusionToNoiseRatio = 1 + r.nextDouble() * 8;     // SNR
//                double diffusionToNoiseRatio = 1 + r.nextDouble() * 15;    // SNR
                double diffusionToNoiseRatio = 3;     // SNR
                double sigmaPosNoise = 1;
                switch (type) {
                    case FREE_rotation:
                        sim = new FreeDiffusionSimulator(D_ND, timelag, dimension, numberOfSteps);
                        sigmaPosNoise = Math.sqrt(D_ND * timelag) / diffusionToNoiseRatio;
                        break;

                    case CONFINED1_TA:
                        double radius_CD_TA = Math.sqrt(BoundednessFeature.a(numberOfSteps) * D_TA * timelag / (4 * boundedness));
                        sim = new ConfinedDiffusionSimulator(D_TA, timelag, radius_CD_TA, dimension, numberOfSteps);
                        sigmaPosNoise = Math.sqrt(D_TA * timelag) / diffusionToNoiseRatio;
                        break;

                    case CONFINED2_TR:
                        double radius_CD_TR = Math.sqrt(BoundednessFeature.a(numberOfSteps) * D_TR * timelag / (4 * boundedness));
                        sim = new ConfinedDiffusionSimulator(D_TR, timelag, radius_CD_TR, dimension, numberOfSteps);
                        sigmaPosNoise = Math.sqrt(D_TR * timelag) / diffusionToNoiseRatio;
                        break;

                    case ACTIVE_DMR:
                        double aToDRatio = 1 + r.nextDouble() * 16; // factor R : 1- 17
//                        double aToDRatio = 5 + r.nextDouble() * 20; // factor R : 5- 25  Nikos Hatzakis's paper parameter
                        double drift = Math.sqrt(aToDRatio * 4 * D_DMR / tracklength);  // speed
                        AbstractSimulator sim1 = new ActiveTransportSimulator(drift, angleVelocity, timelag, dimension, numberOfSteps);
                        AbstractSimulator sim2 = new FreeDiffusionSimulator(D_DMR, timelag, dimension, numberOfSteps);
                        sim = new CombinedSimulator(sim1, sim2);
                        sigmaPosNoise = Math.sqrt(D_DMR * timelag + drift * drift * timelag * timelag) / diffusionToNoiseRatio;
                        break;

                    case ACTIVE_DM:
                        aToDRatio = 1 + r.nextDouble() * 16; // factor R : 1- 17
//                        double aToDRatio = 5 + r.nextDouble() * 20; // factor R : 5- 25  Nikos Hatzakis's paper parameter
                        drift = Math.sqrt(aToDRatio * 4 * D_DM / tracklength);  // speed
                        sim1 = new ActiveTransportSimulator(drift, angleVelocity, timelag, dimension, numberOfSteps);
                        sim2 = new FreeDiffusionSimulator(D_DM, timelag, dimension, numberOfSteps);
                        sim = new CombinedSimulator(sim1, sim2);
                        sigmaPosNoise = Math.sqrt(D_DM * timelag + drift * drift * timelag * timelag) / diffusionToNoiseRatio;
                        break;

                    default:
                        break;
                }
                t = sim.generateTrajectory();
//                if (type == SIM_TYPE.ANOMALOUS) {
//                    t = t.subList(0, numberOfSteps + 1);
//                }

                t.setType(typestring);
                trajectorys.add(t);
                tCounter++;
                if (tCounter % 10 == 0) {
                    System.out.println("T: " + tCounter + " Type " + t.getType());
                }
            }
        }

        System.out.println("Tracks generated");

        try {
            File writeName = new File(path); // Relative path; if it doesn't exist, create a new .txt file.
            if (!writeName.exists()) {
                writeName.createNewFile(); // Create a new file; if a file with the same name exists, overwrite it directly.
            }
            FileWriter writer = new FileWriter(writeName);
            BufferedWriter out = new BufferedWriter(writer);

            out.write('[');
            for (int i = 0; i < numberOfTracks * num_class; i++) {
                out.write(String.valueOf(trajectorys.get(i)));
                if (i != numberOfTracks * num_class - 1) {
                    out.write(',');
                }
            }
            out.write(']');
            out.flush();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Export done");
        System.out.println("Done!");
    }
}
