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

import de.biomedical_imaging.traJ.Trajectory;

import java.util.ArrayList;


/**
 * DOES NOT WORK ANYMORE
 * @author thorsten
 *
 */
public class WindowedClassificationProcess {
	
	public String[] windowedClassification(Trajectory t, AbstractClassifier c, int n){
		int windowsize = 2*n+1;
		String[] types = new String[t.size()];
		for(int i = 0; i < types.length; i++){
			types[i] = "NONE";
		}
	
		ArrayList<Trajectory> tracks = new ArrayList<Trajectory>();
		for(int i = 0; i < (t.size()-windowsize+1);i++){
			Trajectory sub = t.subList(i, i+windowsize-1);
			tracks.add(sub);
		}
		String[] res = c.classify(tracks);
		if(res!=null){
			for(int i = 0; i < res.length; i++){
				types[i + n] = res[i];
			}
		}
		return types;
	}
}
