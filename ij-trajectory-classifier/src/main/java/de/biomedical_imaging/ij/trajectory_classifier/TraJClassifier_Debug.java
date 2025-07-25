/*
MIT License

Copyright (c) 2016 Thorsten Wagner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */


package de.biomedical_imaging.ij.trajectory_classifier;

import de.biomedical_imaging.traJ.DiffusionCoefficientEstimator.RegressionDiffusionCoefficientEstimator;
import de.biomedical_imaging.traJ.Trajectory;
import de.biomedical_imaging.traJ.TrajectoryUtil;
import de.biomedical_imaging.traJ.VisualizationUtils;
import de.biomedical_imaging.traJ.features.*;
import de.biomedical_imaging.traJ.simulation.*;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.CurveFitter;
import ij.process.ByteProcessor;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.knowm.xchart.Chart;
import org.knowm.xchart.Series;
import org.knowm.xchart.SeriesMarker;
import org.knowm.xchart.SwingWrapper;

import java.awt.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.*;

public class TraJClassifier_Debug {

    public static void main(String[] args) {
        CentralRandomNumberGenerator.getInstance().setSeed(10);
        CentralRandomNumberGenerator rand = CentralRandomNumberGenerator.getInstance();
        TraJClassifier_Debug db = new TraJClassifier_Debug();
        Trajectory t = new Trajectory(2);
        for (int i = 0; i < 101; i++) {
            double x = 0.0;
            if (rand.nextDouble() > 0.7) {
                System.out.print("!");
                x = rand.nextDouble();
            }
            t.add(x, 320.0, 0.0);
        }


        String modelpath = "";
        try {
            modelpath = db.ExportResource("/randomForestModel.RData");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        RRFClassifierRenjin c = new RRFClassifierRenjin(modelpath, 1.0 / 10);
        c.start();
        ArrayList<Trajectory> hlp = new ArrayList<Trajectory>();
        hlp.add(t);
        String[] cls = c.classify(hlp);
        System.out.println("Class " + cls[0]);
        //db.analyseConfidence();
		/*
		AbstractSimulator sim =  new StalledSimulator(60, 1);//new AnomalousDiffusionWMSimulation(1, 1, 2, 2000, 0.3);
		double[] val = new double[1000];
		for(int i = 0; i < val.length; i++){
			Trajectory t = sim.generateTrajectory();
		//	t = t.subList(0, 60);
			PowerLawFeature pwf = new PowerLawFeature(t, 1, 1, t.size()/3);
			val[i] = pwf.evaluate()[0];
		}
		Mean m = new Mean();
		StandardDeviation sd = new StandardDeviation();
		Median med = new Median();
		Max max = new Max();
		Min min = new Min();
		Percentile per = new Percentile();
		double perc90 = per.evaluate(val, 95);
		double perc10 = per.evaluate(val, 5);
		System.out.println("Mean: " + m.evaluate(val) + " SD: " + sd.evaluate(val) + " Median: " + med.evaluate(val)+ " Min: " + min.evaluate(val) + " Max: " + max.evaluate(val) + " Perc: " + perc10 + " < alpha <" + perc90);
		*/
    }

    public void analyseConfidence() {
        CentralRandomNumberGenerator.getInstance().setSeed(10);

        String modelpath = "";
        try {
            modelpath = ExportResource("/randomForestModel.RData");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        double diffusioncoefficient = 1;
        double timelag = 1;
        int dimension = 2;
        int N = 100;
        int Nsteps = 800;
        double boundedness = 4;
        double R = 60;
        double drift = Math.sqrt(R * 4 * diffusioncoefficient / 90);
        FreeDiffusionSimulator freeSim = new FreeDiffusionSimulator(diffusioncoefficient, timelag, dimension, Nsteps);
        ActiveTransportSimulator activeSim = new ActiveTransportSimulator(drift, 0, timelag, dimension, Nsteps);
        CombinedSimulator directedSim = new CombinedSimulator(freeSim, activeSim);
        AnomalousDiffusionWMSimulation anomSim = new AnomalousDiffusionWMSimulation(diffusioncoefficient, timelag, dimension, 2000, 0.3);
        Trajectory th = anomSim.generateTrajectory().subList(0, Nsteps);
        Trajectory t = TrajectoryUtil.concactTrajectorie(directedSim.generateTrajectory(), th);
        RRFClassifierRenjin ren = new RRFClassifierRenjin(modelpath, timelag);
        ren.start();
        WeightedWindowedClassificationProcess wwcp = new WeightedWindowedClassificationProcess();
        String[] classes = wwcp.windowedClassification(t, ren, 45, 1);
        ren.stop();
        VisualizationUtils.plotChart(VisualizationUtils.getTrajectoryChart(t));
        double[] conf = wwcp.getPositionConfidence();
        for (int i = 0; i < conf.length; i++) {
            System.out.println((i + 1) + "\t" + conf[i] + "\t" + classes[i]);
        }

    }

    public void testSubsampling() {
        CentralRandomNumberGenerator.getInstance().setSeed(10);
        String modelpath = "";
        try {
            modelpath = ExportResource("/randomForestModel.RData");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        double diffusioncoefficient = 50;
        double SNR = 5;
        int n = 1;
        double sigma = Math.sqrt(diffusioncoefficient * 1.0 / 30) / SNR;
        FreeDiffusionSimulator freesim = new FreeDiffusionSimulator(diffusioncoefficient, 1.0 / 30, 2, 250);
        int N = 2000;
        RRFClassifierRenjin cl = new RRFClassifierRenjin(modelpath, 1.0 / 30);

        cl.start();
        ArrayList<Trajectory> tracks = new ArrayList<Trajectory>();
        for (int i = 0; i < N; i++) {
            Trajectory t = freesim.generateTrajectory();
            t = SimulationUtil.addPositionNoise(t, sigma);
            t = TrajectoryUtil.resample(t, n);
            //System.out.println("S: "+ t.size());
            tracks.add(t);
        }
        cl.setTimelag(n * 1.0 / 30);
        String[] res = cl.classify(tracks);

        int scounter = 0;
        int ncounter = 0;
        for (String s : res) {
            if (s.equals("SUBDIFFUSION")) {
                scounter++;
            }
            if (s.equals("NORM. DIFFUSION")) {
                ncounter++;
            }
        }
        System.out.println("SC: " + (1.0 * scounter) / res.length);
        System.out.println("SN: " + (1.0 * ncounter) / res.length);
    }


    public static double meanAbsoluteSteplength(Trajectory t) {
        double[] v = new double[t.size() - 1];

        for (int i = 1; i < t.size(); i++) {
            v[i - 1] = Math.abs(t.get(i).distance(t.get(i - 1)));
        }

        Mean m = new Mean();

        return m.evaluate(v);
    }

    public static void importTracksAndShow() {
        new ImageJ();
        IJ.getInstance().show(true);
        ImageStack is = new ImageStack(1000, 1000);
        for (int i = 0; i < 1005; i++) {
            is.addSlice(new ByteProcessor(1000, 1000));
        }

        ImagePlus img = new ImagePlus("", is);
        img.show();


        TraJClassifier_ tclass = new TraJClassifier_();
        tclass.run("");
    }

    public void showProblematicTrack() {
        String modelpath = "";
        try {
            modelpath = ExportResource("/randomForestModel.RData");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        ExportImportTools eit = new ExportImportTools();
        ArrayList<Trajectory> tracks = eit.importTrajectoryDataFromCSV("/home/thorsten/subdiffusion_alpha.csv");
        Trajectory t = tracks.get(0);
        PowerLawFeature pwf = new PowerLawFeature(t, 10, 1, t.size() / 3);
        double[] ev = pwf.evaluate();
        double a = ev[0];
        double D = ev[1];
        Chart c = VisualizationUtils.getMSDLineWithPowerModelChart(t, 1, t.size() / 3, 1.0 / 10, 0.4, D * 10);
        System.out.println("alpha: " + a);
        VisualizationUtils.plotChart(c);
    }

    public void showTestScene() {
        CentralRandomNumberGenerator.getInstance().setSeed(2);
        String modelpath = "";
        try {
            modelpath = ExportResource("/randomForestModel.RData");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        int w = 45;

        double diffusioncoefficient = 0.05;
        double timelag = 1.0 / 30;
        double[] driftspeed = {0, 0.27, 1, 2.4}; // µm/s
        double angularVelocity = Math.PI / 4; //rad/s
        int numberOfSteps = 500;
        int diffusionToNoiseRatio = 1;
        double sigmaPosNoise = Math.sqrt(2 * diffusioncoefficient * timelag) / diffusionToNoiseRatio;

        double SNR = 2.5;
        double boundedness = 5;
        double radius_confined = Math.sqrt(BoundednessFeature.a(w) * diffusioncoefficient * timelag / (4 * boundedness));
        System.out.println("Radius confined" + radius_confined + " µm");
        double R = 5;
        double velocity = Math.sqrt(R * 4 * diffusioncoefficient / (w * timelag));
        System.out.println("Velocity " + velocity + " µm/s");

        double Dc = diffusioncoefficient;
        double rsig_1 = Math.sqrt(Dc * timelag) / SNR;
        double rsig_2 = Math.sqrt(Dc * timelag + velocity * velocity * timelag * timelag) / SNR;
        System.out.println("Sigma 1: " + rsig_1 + " Sigma 2: " + rsig_2);

        ArrayList<Trajectory> orig = new ArrayList<Trajectory>();
        //FREE -> Active -> Confined -> Active -> Anomalous
        Trajectory combinedTrajectory = new Trajectory(2);

        AbstractSimulator conf = new ConfinedDiffusionSimulator(diffusioncoefficient, timelag, radius_confined, 2, 200);
        Trajectory tConf = conf.generateTrajectory();
        tConf = SimulationUtil.addPositionNoise(tConf, rsig_1);
        tConf.setType("confined");
        orig.add(tConf);

        AbstractSimulator simFree = new FreeDiffusionSimulator(diffusioncoefficient, timelag, 2, 330);
        ArrayList<Trajectory> t = new ArrayList<Trajectory>();
        Trajectory trFree = simFree.generateTrajectory();
        trFree = SimulationUtil.addPositionNoise(trFree, rsig_1);
        trFree.setType("normal");
        trFree.offset(tConf.get(tConf.size() - 1).x, tConf.get(tConf.size() - 1).y, 0);
        orig.add(trFree);
        combinedTrajectory = TrajectoryUtil.concactTrajectorie(tConf, trFree);


        simFree = new FreeDiffusionSimulator(diffusioncoefficient, timelag, 2, 200);

        ActiveTransportSimulator simActive = new ActiveTransportSimulator(velocity, angularVelocity, 1.51 * Math.PI, timelag, 2, 200);
        Trajectory trDirected = TrajectoryUtil.combineTrajectory(simActive.generateTrajectory(), simFree.generateTrajectory());
        trDirected = SimulationUtil.addPositionNoise(trDirected, rsig_2);
        combinedTrajectory = TrajectoryUtil.concactTrajectorie(combinedTrajectory, trDirected);
        trDirected.setType("directed");
        trDirected.offset(trFree.get(trFree.size() - 1).x, trFree.get(trFree.size() - 1).y, 0);
        orig.add(trDirected);

        AbstractSimulator simAnom = new AnomalousDiffusionWMSimulation(diffusioncoefficient, timelag, 2, 200, 0.5);
        Trajectory trAnom = simAnom.generateTrajectory();
        trAnom = SimulationUtil.addPositionNoise(trAnom, rsig_1);
        trAnom.setType("anom");
        trAnom.offset(trDirected.get(trDirected.size() - 1).x, trDirected.get(trDirected.size() - 1).y, 0);
        orig.add(trAnom);
        combinedTrajectory = TrajectoryUtil.concactTrajectorie(combinedTrajectory, trAnom);
		/*
		Trajectory trDirected2 = TrajectoryUtil.combineTrajectory(simActive.generateTrajectory(), simFree.generateTrajectory());
		trDirected2 = SimulationUtil.addPositionNoise(trDirected2, rsig_2);
		combinedTrajectory = TrajectoryUtil.concactTrajectorie(combinedTrajectory, trDirected2);
		
		
		*/
        //	Chart c = VisualizationUtils.getTrajectoryChart(combinedTrajectory);
        //VisualizationUtils.plotChart(c);
        ExportImportTools eit = new ExportImportTools();

        ArrayList<Subtrajectory> tracks = TraJClassifier_.getInstance().classifyAndSegment(combinedTrajectory, modelpath, w, 90, 10, 1);
        for (Trajectory tr : tracks) {
            tr.offset(2.0, 0, 0);
        }
        eit.exportTrajectoryDataAsCSV(tracks, "/home/thorsten/track_seg.csv");
        showTracks(tracks);

        eit.exportTrajectoryDataAsCSV(orig, "/home/thorsten/track_orig.csv");
        showTracks(orig);
		/*
		RRFClassifierRenjin rrf = new RRFClassifierRenjin(modelpath,timelag);
		rrf.start();
		WeightedWindowedClassificationProcess wcp = new WeightedWindowedClassificationProcess();
		w=60;
		String[] classes = wcp.windowedClassification(combinedTrajectory, rrf, w,1);
		double[] conf = wcp.getPositionConfidence();
		for(int i = 0; i < conf.length; i++){
			//System.out.println("i: " +(i+1)+ " conf: " + conf[i]);
		}
		Mean mean2 = new Mean();
		System.out.println(" Mean conf: " + mean2.evaluate(conf));
		//classes=movingMode(classes, 10);
		//classes = movingMode(classes, w/2);
		//showTrack(combinedTrajectory, classes,0);
		segmentSubtrajectoriesAndShow(combinedTrajectory, classes, conf,0,10);
		*/
    }


    public static void outputFeatures(Trajectory t, double timelag) {
        int numberOfSegmentsSplineFit = 7;
        int numberOfPointsForShortTimeLongTimeRatio = 2;
        AbstractTrajectoryFeature f = new ElongationFeature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new FractalDimensionFeature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new MeanSquaredDisplacmentCurvature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new RegressionDiffusionCoefficientEstimator(t, 1.0 / timelag, 1, 3);
        PowerLawFeature f2 = new PowerLawFeature(t, 1 / timelag, 1, t.size() / 3, 0.5, f.evaluate()[0]);
        System.out.println(f.getShortName() + ": " + f2.evaluate()[0]);

        //	StandardDeviationDirectionFeature sdf = new StandardDeviationDirectionFeature(t, timelagForDirectionDeviationLong);
        //	pool.submit(new FeatureWorker(sdDir, i,sdf, EVALTYPE.FIRST));
        //	if(chatty)System.out.println("SDDIR evaluated");

        f = new SplineCurveDynamicsFeature(t, numberOfSegmentsSplineFit, 1);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0] / f.evaluate()[1]);

        f = new AsymmetryFeature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);


        f = new Asymmetry2Feature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new Asymmetry3Feature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new EfficiencyFeature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new ShortTimeLongTimeDiffusioncoefficentRatio(t, numberOfPointsForShortTimeLongTimeRatio);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new KurtosisFeature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new SkewnessFeature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new MSDRatioFeature(t, 1, 5);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new StraightnessFeature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new TrappedProbabilityFeature(t);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);

        f = new GaussianityFeauture(t, 1);
        System.out.println(f.getShortName() + ": " + f.evaluate()[0]);
    }


    public static double[] evaluate(Trajectory t, double timelag) {
        MeanSquaredDisplacmentFeature msd = new MeanSquaredDisplacmentFeature(t, 1);
        msd.setOverlap(false);

        ArrayList<Double> xDataList = new ArrayList<Double>();
        ArrayList<Double> yDataList = new ArrayList<Double>();
        for (int i = 1; i < t.size(); i++) {
            msd.setTimelag(i);
            double[] res = msd.evaluate();
            double msdvalue = res[0];
            int N = (int) res[2];
            for (int j = 0; j < N; j++) {
                xDataList.add((double) i * timelag);
                yDataList.add(msdvalue);
            }
        }
        double[] xData = ArrayUtils.toPrimitive(xDataList.toArray(new Double[0]));
        double[] yData = ArrayUtils.toPrimitive(yDataList.toArray(new Double[0]));
        CurveFitter fitter = new CurveFitter(xData, yData);
        MaxDistanceBetweenTwoPositionsFeature maxdist = new MaxDistanceBetweenTwoPositionsFeature(t);
        RegressionDiffusionCoefficientEstimator regest = new RegressionDiffusionCoefficientEstimator(t, 1 / timelag, 5, 5);
        double estrad = maxdist.evaluate()[0];
        double estDC = regest.evaluate()[0];
        double[] initialParams = {estrad * estrad};//,regest.evaluate()[0]};
        //fitter.doCustomFit("y=a*(1-b*exp(-4*c*"+estDC+"*x/a))", initialParams, false);
        fitter.doCustomFit("y=a*(1-exp(-4*" + estDC + "*x/a))", initialParams, false);
        double[] params = fitter.getParams();
        double[] res = {params[0]};//params[1],params[2],fitter.getFitGoodness()};
        return res;
    }

    public static String[] windowedClassification(Trajectory t, AbstractClassifier c, int n) {
        int windowsize = 2 * n + 1;
        String[] types = new String[t.size()];
        for (int i = 0; i < n; i++) {
            types[i] = "NONE";
        }
        for (int i = types.length - n; i < types.length; i++) {
            types[i] = "NONE";
        }
        ArrayList<Trajectory> tracks = new ArrayList<Trajectory>();
        for (int i = 0; i < (t.size() - windowsize + 1); i++) {
            Trajectory sub = t.subList(i, i + windowsize - 1);
            tracks.add(sub);
        }

        String[] res = c.classify(tracks);
        for (int i = 0; i < res.length; i++) {
            types[i + n] = res[i];
        }

        return types;
    }

    public ArrayList<Trajectory> segmentSubtrajectoriesAndShow(Trajectory t, String[] classes, double[] classConfidence, double confidenceThreshold, int minSegLength) {

        //segmentation
        if (t.size() != classes.length) {
            throw new IllegalArgumentException("Tracklength and the number of classes have to be equal");
        }
        ArrayList<String> cls = new ArrayList<String>();
        ArrayList<Trajectory> subtracks = new ArrayList<Trajectory>();
        String prevCls = classes[0];
        Trajectory sub = new Trajectory(t.getDimension());
        sub.add(t.get(0));
        for (int i = 1; i < t.size(); i++) {
            if (prevCls == classes[i]) {
                sub.add(t.get(i));
            } else {

                subtracks.add(sub);
                cls.add(prevCls);

                prevCls = classes[i];
                sub = new Trajectory(t.getDimension());
                sub.add(t.get(i));
            }
        }

        cls.add(prevCls);
        subtracks.add(sub);

        //Confidence filter;
        int indexCounter = 0;
        for (int i = 0; i < subtracks.size(); i++) {
            for (int j = 0; j < subtracks.get(i).size(); j++) {
                if (classConfidence[indexCounter] < confidenceThreshold) {
                    subtracks.get(i).remove(j);
                    j--;
                }
                indexCounter++;
            }
        }

        //Min Segmenth length filter;
        for (int i = 0; i < subtracks.size(); i++) {
            if (subtracks.get(i).size() < minSegLength) {
                subtracks.remove(i);
                cls.remove(i);
                i--;

            }
        }
        showTracks(subtracks, cls);
        return subtracks;

    }

    public static void showTrack(Trajectory t, String[] classes, int minSegLength) {
        if (t.size() != classes.length) {
            throw new IllegalArgumentException("Tracklength and the number of classes have to be equal");
        }
        ArrayList<String> cls = new ArrayList<String>();
        ArrayList<Trajectory> subtracks = new ArrayList<Trajectory>();
        String prevCls = classes[0];
        Trajectory sub = new Trajectory(t.getDimension());
        sub.add(t.get(0));
        for (int i = 1; i < t.size(); i++) {
            if (prevCls == classes[i]) {
                sub.add(t.get(i));
            } else {
                if (sub.size() > minSegLength) {
                    subtracks.add(sub);
                    cls.add(prevCls);
                }

                prevCls = classes[i];
                sub = new Trajectory(t.getDimension());
                sub.add(t.get(i));
            }
        }

        if (sub.size() > minSegLength) {
            cls.add(prevCls);
            subtracks.add(sub);
        }
        showTracks(subtracks, cls);
    }

    public static void showTracks(ArrayList<? extends Trajectory> tracks) {
        ArrayList<String> cls = new ArrayList<String>();
        for (int i = 0; i < tracks.size(); i++) {
            cls.add(tracks.get(i).getType());
        }
        showTracks(tracks, cls);
    }

    public static void showTracks(ArrayList<? extends Trajectory> tracks, String[] classes) {
        ArrayList<String> cls = new ArrayList<String>();
        for (int i = 0; i < classes.length; i++) {
            cls.add(classes[i]);
        }
        showTracks(tracks, cls);
    }


    public static void showTracks(ArrayList<? extends Trajectory> tracks, ArrayList<String> classes) {
        Color[] colors = new Color[]
                {
                        Color.RED, Color.BLUE, Color.YELLOW, Color.GREEN,
                        Color.BLACK, Color.PINK, Color.ORANGE, Color.MAGENTA
                };
        HashSet<String> uniqueClasses = new HashSet<String>();
        uniqueClasses.addAll(classes);
        HashMap<String, Integer> mapTypeToInt = new HashMap<String, Integer>();
        HashMap<Integer, String> mapIntToType = new HashMap<Integer, String>();
        Iterator<String> it = uniqueClasses.iterator();
        int key = 0;
        while (it.hasNext()) {
            String type = it.next();
            mapTypeToInt.put(type, key);
            mapIntToType.put(key, type);
            key++;
        }

        Chart chart = new Chart(700, 400);
        for (int i = 0; i < tracks.size(); i++) {
            Trajectory t = tracks.get(i);
            String type = classes.get(i);
            if (t.getDimension() == 2) {
                double[] xData = new double[t.size()];
                double[] yData = new double[t.size()];
                for (int j = 0; j < t.size(); j++) {
                    xData[j] = t.get(j).x;
                    yData[j] = t.get(j).y;

                }
                // Create Chart
                Series s = chart.addSeries(i + ". " + type + "(" + t.size() + ")", xData, yData);
                s.setLineColor(colors[mapTypeToInt.get(type)]);
                s.setMarker(SeriesMarker.NONE);


            }
        }

        //Show it
        new SwingWrapper(chart).displayChart();
    }

    public static void printTypes(String[] res) {
        String prev = res[0];
        int c = 1;
        for (int i = 1; i < res.length; i++) {
            if (prev == res[i]) {
                c++;
            } else {
                System.out.println("Type: " + prev + "(" + c + ")");
                c = 1;
                prev = res[i];
            }

        }
        System.out.println("Type: " + prev + "(" + c + ")");
    }

    public static void addToTypes(ArrayList<String> types, String type, int n) {

        for (int i = 0; i < n; i++) {
            types.add(type);
        }
    }

    public static String[] movingMode(String[] types, int n) {
        ArrayList<String> ltypes = new ArrayList<String>();
        for (int i = 0; i < types.length; i++) {
            ltypes.add(types[i]);
        }
        return movingMode(ltypes, n);
    }

    public static String[] movingMode(ArrayList<String> types, int n) {
        int windowsize = 2 * n + 1;
        HashSet<String> uniqueTypes = new HashSet<String>();
        uniqueTypes.addAll(types);
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

        String[] medTypes = new String[types.size()];

        for (int i = 0; i < n; i++) {
            medTypes[i] = types.get(i);
        }
        for (int i = types.size() - n; i < types.size(); i++) {
            medTypes[i] = types.get(i);
        }

        for (int i = 0; i < (types.size() - windowsize + 1); i++) {
            List<String> sub = types.subList(i, i + windowsize - 1);
            double[] values = new double[sub.size()];
            for (int j = 0; j < sub.size(); j++) {
                values[j] = mapTypeToInt.get(sub.get(j));
            }

            medTypes[i + n] = mapIntToType.get(((int) StatUtils.mode(values)[0]));
        }
        return medTypes;
    }

    /**
     * Export a resource embedded into a Jar file to the local file path.
     *
     * @param resourceName ie.: "/SmartLibrary.dll"
     * @return The path to the exported resource
     * @throws Exception
     */
    public String ExportResource(String resourceName) throws Exception {
        InputStream stream = null;
        OutputStream resStreamOut = null;
        String tmpFolder;
        try {
            stream = this.getClass().getResourceAsStream(resourceName);//note that each / is a directory down in the "jar tree" been the jar the root of the tree
            if (stream == null) {
                IJ.error("Cannot get resource \"" + resourceName + "\" from Jar file.");
                throw new Exception("Cannot get resource \"" + resourceName + "\" from Jar file.");
            }

            int readBytes;
            byte[] buffer = new byte[4096];
            File folderDir = new File(IJ.getDirectory("temp") + "/.trajclassifier");

            // if the directory does not exist, create it
            if (!folderDir.exists()) {
                folderDir.mkdir();
            }
            tmpFolder = folderDir.getPath().replace('\\', '/');
            resStreamOut = new FileOutputStream(tmpFolder + resourceName);
            while ((readBytes = stream.read(buffer)) > 0) {
                resStreamOut.write(buffer, 0, readBytes);
            }
        } catch (Exception ex) {
            IJ.error(ex.getMessage());
            throw ex;
        } finally {
            stream.close();
            resStreamOut.close();
        }

        return tmpFolder + resourceName;
    }


}
