package de.biomedical_imaging.ij.trajectory_classifier;

import ij.IJ;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

import de.biomedical_imaging.traJ.Trajectory;
import de.biomedical_imaging.traJ.TrajectoryUtil;

public class WeightedWindowedClassificationProcess {

    private double[] posConfidence = null;
    private double numberOmittedSegments = 0;

    /**
     * @param t    Trajectory to be classified
     * @param c    Classifier
     * @param n    Determines the window size w = 2*n + 1
     * @param rate Resampling rate (1=no resampling)
     * @return An array with the class for each position.
     */
    public String[] windowedClassification(Trajectory t, AbstractClassifier c, int n, int rate) {
        int windowsize = 2 * n + 1;

        int increment = 1;
        ArrayList<Trajectory> tracks = new ArrayList<Trajectory>();
        for (int i = 0; i < (t.size() - windowsize + increment); i = i + increment) {
            Trajectory sub = t.subList(i, i + windowsize - 1);
            if (rate > 1) {
                sub = TrajectoryUtil.resample(sub, rate);
            }
            tracks.add(sub);
        }
        String[] res = c.classify(tracks);
        double[] confidence = c.getConfindence();
        for (int i = 0; i < res.length; i++) {
            if (res[i] == null) {
                return null;
            }
        }

        String[] types = applyWeightening(res, confidence, n, t.size());
        return types;
    }


    public double[] getPositionConfidence() {
        return posConfidence;
    }

    protected String[] applyWeightening(String[] res, double[] confidence, int n, int tracklength) {

        String[] types = new String[tracklength];
        posConfidence = new double[tracklength];
        //	for(int i = 0; i < types.length; i++){
        //		types[i] = "NONE";
        //	}


        //Build mapping
        ArrayList<String> restypes = new ArrayList<String>();
        for (int i = 0; i < res.length; i++) {
            restypes.add(res[i]);
        }
        //	restypes.add("NONE");
        HashSet<String> uniqueTypes = new HashSet<String>();
        uniqueTypes.addAll(restypes);
        HashMap<String, Integer> mapTypeToInt = new HashMap<String, Integer>();
        HashMap<Integer, String> mapIntToType = new HashMap<Integer, String>();
        Iterator<String> it = uniqueTypes.iterator();
        int key = 0;
        while (it.hasNext()) {
            String type = it.next();
            mapTypeToInt.put(type, key);
            mapIntToType.put(key, type);
            key++;
        }

        ArrayList<Double[]> weightes = new ArrayList<Double[]>();
        ArrayList<Integer[]> Nvotes = new ArrayList<Integer[]>();
        for (int i = 0; i < tracklength; i++) {
            Double[] h = new Double[key];
            Arrays.fill(h, new Double(0));
            weightes.add(h);

            Integer[] h1 = new Integer[key];
            Arrays.fill(h1, new Integer(0));
            Nvotes.add(h1);
        }


        for (int i = 0; i < res.length; i++) {
            for (int j = i; j < i + 2 * n + 1; j++) {
                int typ = mapTypeToInt.get(res[i]);

                try {
                    weightes.get(j)[typ] = weightes.get(j)[typ] + confidence[i];
                    Nvotes.get(j)[typ] = Nvotes.get(j)[typ] + 1;
                } catch (Exception e) {

                    IJ.log("Res: " + res[i] + " j: " + j + " i: " + i + " weigthes size " + weightes.size());
                }
            }
        }

        for (int i = 0; i < types.length; i++) {
            if (weightes.get(i).length > 0) {
                double[] result = getHighest(weightes.get(i));
                int mode1 = (int) result[0];
                double maxv = result[1];
                double wConf = maxv / Nvotes.get(i)[mode1];
                String mode = mapIntToType.get(mode1);
                types[i] = mode;
                posConfidence[i] = wConf;

            }
            //else{
            //	types[i] = "NONE";
            //}
        }
        return types;
    }

    private double[] getHighest(Double[] weightes) {
        double max = 0;
        int maxindex = 0;
        for (int i = 0; i < weightes.length; i++) {
            if (weightes[i] > max) {
                max = weightes[i];
                maxindex = i;
            }
        }
        return new double[]{maxindex, max};
    }


    public double[] arrListToArray(ArrayList<Integer> l) {
        double[] a = new double[l.size()];
        for (int i = 0; i < l.size(); i++) {
            a[i] = l.get(i).intValue();
        }
        return a;
    }

    public int[] arrListToArrayInt(ArrayList<Integer> l) {
        int[] a = new int[l.size()];
        for (int i = 0; i < l.size(); i++) {
            a[i] = l.get(i).intValue();
        }
        return a;
    }
}
