#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <string.h>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <map>
//#include "loader.h"
using namespace std;

// implementation of k-NN by Wang et al.

//Data parameters
int FEAT_NUM = 0; //number of features
int ROUND_NUM = 5000000;
int NEIGHBOR_NUM; //number of neighbors for kNN
int RECOPOINTS_NUM = 5; //number of neighbors for distance learning
int TRAIN_CLOSED_SITENUM, TRAIN_CLOSED_INSTNUM, TRAIN_OPEN_INSTNUM,
    TEST_CLOSED_SITENUM, TEST_CLOSED_INSTNUM, TEST_OPEN_INSTNUM;
bool OPEN_MAJORITY = true;
bool open_world;
map<string, string> d;

bool inarray(int ele, int* array, int len) {
	for (int i = 0; i < len; i++) {
		if (array[i] == ele)
			return 1;
	}
	return 0;
}

void alg_init_weight(float* weight) {
	for (int i = 0; i < FEAT_NUM; i++) {
		weight[i] = (rand() % 100) / 100.0 + 0.5;
	}
}

float dist(float* feat1, float* feat2, float* weight) {
	float toret = 0;
	for (int i = 0; i < FEAT_NUM; i++) {
		if (feat1[i] != -1 and feat2[i] != -1) {
			toret += weight[i] * abs(feat1[i] - feat2[i]);
		}
	}
	return toret;
}

void alg_recommend2(float** feat, int* featclasses, int featlen, float* weight) {
	float* dist_list = new float[featlen];
	int* recogoodlist = new int[RECOPOINTS_NUM];
	int* recobadlist = new int[RECOPOINTS_NUM];

    //int num_weights = ROUND_NUM/featlen; // original value; depends on the number of files, bad results when too many files (repeated protected traces)
    int num_weights = 832; // performance very dependent on this value, this one gives the best results (this is the result of ROUND_NUM/featlen with only original traces)
	for (int i = 0; i < num_weights; i++) {
		int id = i % featlen;
		printf("\rLearning weights... %d (%d-%d)", i, 0, num_weights-1);
		fflush(stdout);

		int true_class = featclasses[id];
/*
		int cur_site, cur_inst;
		if (id < CLOSED_SITENUM * CLOSED_INSTNUM) {
			cur_site = id/CLOSED_INSTNUM;
			cur_inst = id % CLOSED_INSTNUM;
		}
		else {
			cur_site = CLOSED_SITENUM;
			cur_inst = id - CLOSED_SITENUM * CLOSED_INSTNUM;
		}
*/

		//learn distance to other feat elements, put in dist_list
		for (int k = 0; k < featlen; k++) {
			dist_list[k] = dist(feat[id], feat[k], weight);
		}
		//set my own distance to max
		float max = *max_element(dist_list, dist_list+featlen);
		dist_list[id] = max;

		float pointbadness = 0;
		float maxgooddist = 0; //the greatest distance of all the good neighbours NEIGHBOR_NUM

		//find the good neighbors: NEIGHBOR_NUM lowest dist_list values of the same class
		for (int k = 0; k < RECOPOINTS_NUM; k++) {
			int minind; //ind of minimum element of dist_list
			float mindist = max;
			for (int dind = 0; dind < featlen; dind++) {
				if (featclasses[dind] == true_class and dist_list[dind] < mindist) {
					minind = dind;
					mindist = dist_list[dind];
				}
			}
			if (dist_list[minind] > maxgooddist) maxgooddist = dist_list[minind];
			dist_list[minind] = max;
			recogoodlist[k] = minind;
		}
		for (int dind = 0; dind < featlen; dind++) {
			if (featclasses[dind] == true_class) {
				dist_list[dind] = max;
			}
		}
		for (int k = 0; k < RECOPOINTS_NUM; k++) {
			int ind = min_element(dist_list, dist_list+featlen) - dist_list;
			if (dist_list[ind] <= maxgooddist) pointbadness += 1;
			dist_list[ind] = max;
			recobadlist[k] = ind;
		}

		pointbadness /= float(RECOPOINTS_NUM);
		pointbadness += 0.2;
		/*
		if (i == 0) {
			float gooddist = 0;
			float baddist = 0;
			printf("Current point: %d\n", i);
			printf("Bad points:\n");
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				printf("%d, %f\n", recobadlist[k], dist(feat[i], feat[recobadlist[k]], weight));
				baddist += dist(feat[i], feat[recobadlist[k]], weight);
			}

			printf("Good points:\n");
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				printf("%d, %f\n", recogoodlist[k], dist(feat[i], feat[recogoodlist[k]], weight));
				gooddist += dist(feat[i], feat[recogoodlist[k]], weight);
			}

			printf("Total bad distance: %f\n", baddist);
			printf("Total good distance: %f\n", gooddist);
		}*/

		float* featdist = new float[FEAT_NUM];
		for (int f = 0; f < FEAT_NUM; f++) {
			featdist[f] = 0;
		}
		int* badlist = new int[FEAT_NUM];
		int minbadlist = 0;
		int countbadlist = 0;
		//printf("%d ", badlist[3]);
		for (int f = 0; f < FEAT_NUM; f++) {
			if (weight[f] == 0) badlist[f] = 0;
			else {
			float maxgood = 0;
			int countbad = 0;
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				float n = abs(feat[id][f] - feat[recogoodlist[k]][f]);
				if (feat[id][f] == -1 or feat[recogoodlist[k]][f] == -1)
					n = 0;
				if (n >= maxgood) maxgood = n;
			}
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				float n = abs(feat[id][f] - feat[recobadlist[k]][f]);
				if (feat[id][f] == -1 or feat[recobadlist[k]][f] == -1)
					n = 0;
				//if (f == 3) {
				//	printf("%d %d %f %f\n", i, k, n, maxgood);
				//}
				featdist[f] += n;
				if (n <= maxgood) countbad += 1;
			}
			badlist[f] = countbad;
			if (countbad < minbadlist) minbadlist = countbad;
			}
		}

		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] != minbadlist) countbadlist += 1;
		}
		int* w0id = new int[countbadlist];
		float* change = new float[countbadlist];

		int temp = 0;
		float C1 = 0;
		float C2 = 0;
		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] != minbadlist) {
				w0id[temp] = f;
				change[temp] = weight[f] * 0.02 * badlist[f]/float(RECOPOINTS_NUM); //* pointbadness;
				//if (change[temp] < 1.0/1000) change[temp] = weight[f];
				C1 += change[temp] * featdist[f];
				C2 += change[temp];
				weight[f] -= change[temp];
				temp += 1;
			}
		}

		/*if (i == 0) {
			printf("%d %f %f\n", countbadlist, C1, C2);
			for (int f = 0; f < 30; f++) {
				printf("%f %f\n", weight[f], featdist[f]);
			}
		}*/
		float totalfd = 0;
		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] == minbadlist and weight[f] > 0) {
				totalfd += featdist[f];
			}
		}

		for (int f = 0; f < FEAT_NUM; f++) {
			if (badlist[f] == minbadlist and weight[f] > 0) {
				weight[f] += C1/(totalfd);
			}
		}

		/*if (i == 0) {
			printf("%d %f %f\n", countbadlist, C1, C2);
			for (int f = 0; f < 30; f++) {
				printf("%f %f\n", weight[f], featdist[f]);
			}
		}*/

		/*if (i == 0) {
			float gooddist = 0;
			float baddist = 0;
			printf("Current point: %d\n", i);
			printf("Bad points:\n");
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				printf("%d, %f\n", recobadlist[k], dist(feat[i], feat[recobadlist[k]], weight));
				baddist += dist(feat[i], feat[recobadlist[k]], weight);
			}

			printf("Good points:\n");
			for (int k = 0; k < RECOPOINTS_NUM; k++) {
				printf("%d, %f\n", recogoodlist[k], dist(feat[i], feat[recogoodlist[k]], weight));
				gooddist += dist(feat[i], feat[recogoodlist[k]], weight,);
			}

			printf("Total bad distance: %f\n", baddist);
			printf("Total good distance: %f\n", gooddist);
		}*/

		delete[] featdist;
		delete[] w0id;
		delete[] change;
		delete[] badlist;
	}


	/*for (int j = 0; j < FEAT_NUM; j++) {
		if (weight[j] > 0)
			weight[j] *= (0.9 + (rand() % 100) / 500.0);
	}*/
	printf("\n");
	delete[] dist_list;
	delete[] recobadlist;
	delete[] recogoodlist;



}

void accuracy(float** trainfeat, float** testfeat, int* trainfeatclasses, int* testfeatclasses, int* testfeatindices, int trainlen, int testlen, float* weight) {
	float* dist_list = new float[trainlen];

	printf("trainlen %d testlen %d\n", trainlen, testlen);

	int tp = 0;
	int fp = 0;
	int p = 0;
	int n = 0;

	FILE * flog;
	cout << "Writing in " << d["RESULT_FILE"] << endl;
	flog = fopen(d["RESULT_FILE"].c_str(), "a");

	for (int is = 0; is < testlen; is++) {

		int true_class = testfeatclasses[is];
		int index = testfeatindices[is];


		map<int, int> class_list;
		printf("\rComputing scores for test files with %d trains... %d (%d-%d)", trainlen, is, 0, testlen-1);
		fflush(stdout);
		for (int at = 0; at < trainlen; at++) {
			dist_list[at] = dist(testfeat[is], trainfeat[at], weight);
		}
		float max = *max_element(dist_list, dist_list+trainlen);

		//log the match score of each class

		fprintf(flog, "%d_%d", true_class, index);

		int CLASS_NUM = atoi(d["CLOSED_SITENUM"].c_str());
		if (open_world) CLASS_NUM += 1;

		map<int, float> match;
		for (int i = 0; i < CLASS_NUM; i++) {
			match[i] = max;
		}

		if (open_world) {

            int guess_class = 0;
            int max_class = 0;

            for (int i = 0; i < NEIGHBOR_NUM; i++) {
                int ind = find(dist_list, dist_list + trainlen, *min_element(dist_list, dist_list+trainlen)) - dist_list;
                int class_ind = trainfeatclasses[ind];
                if(class_list.find(class_ind) == class_list.end()) class_list[class_ind] = 1;
                else class_list[class_ind] += 1;

                if (class_list[class_ind] > max_class) {
                    max_class = class_list[class_ind];
                    guess_class = class_ind;
                }
                dist_list[ind] = max;
            }

            if (OPEN_MAJORITY) {
                bool has_consensus = false;
                // if strict majority agrees
                if (class_list[guess_class] > NEIGHBOR_NUM/2) has_consensus = true;

                if (!has_consensus) {
                    guess_class = -1;
                }
            }
            if (guess_class != -1) {
                if (true_class == guess_class) tp += 1;
                else fp += 1;
            }
            if (index == -1) n += 1;
            else p += 1;

            // write in file for computing accuracy with data consistent with closed world
            for (int i = 0; i < CLASS_NUM; i++) {
                float val = 0;
                if(i == guess_class || (guess_class == -1 && i == CLASS_NUM-1)) {
                    // write 1 for guessed class
                    val = 1;
                }
                fprintf(flog, "\t%f", val);
            }
            fprintf(flog, "\n");
            fflush(flog);
		}
		else {
            for (int at = 0; at < trainlen; at++) {
                int class_ind = trainfeatclasses[at];
                if (class_ind == -1) class_ind = CLASS_NUM-1;
                if (dist_list[at] < match[class_ind]) match[class_ind] = dist_list[at];
            }

            //additive inverse is match
            for (int i = 0; i < CLASS_NUM; i++) {
                match[i] = max - match[i];
                fprintf(flog, "\t%f", match[i]);
            }
            fprintf(flog, "\n");
            fflush(flog);
		}

	}

//	if (open_world) {
//        printf("\ntp:%d fp:%d p:%d n:%d\n", tp, fp, p, n);
//        //fprintf(flog, "tp:%d fp:%d p:%d n:%d\n", tp, fp, p, n);
//
//        printf("Accuracy is %f\n", ((float) tp)/((float) p));
//        //fprintf(flog, "Accuracy is %f\n", ((float) tp)/((float) p));
//	}

    printf("\n");
	fclose(flog);

    cout << "Accuracy computed" << endl;

	delete[] dist_list;
}

//reads fname (a file name) for a single file
void read_feat(float* feat, string fname) {
	ifstream fread;
	fread.open(fname.c_str());
	string str = "";
	getline(fread, str);
	fread.close();

	string tempstr = "";
	int feat_count = 0;
	for (int i = 0; i < str.length(); i++) {
		if (str[i] == ' ') {
			if (tempstr.c_str()[1] == 'X') {
				feat[feat_count] = -1;
			}
			else {
				feat[feat_count] = atof(tempstr.c_str());
			}
			feat_count += 1;
			tempstr = "";
		}
		else {
			tempstr += str[i];
		}
	}
}

void read_filelist(float ** feat, int * featclasses, int *featindices, int featlen, string fname) {
	ifstream fread;
	fread.open(fname.c_str());

	int readcount = 0;
	while (fread.peek() != EOF) {
		string str = "";
		string rstr = "";
		getline(fread, rstr);
		int found = rstr.find_last_of("/");
		str = rstr.substr(found+1);
		str = str.substr(0, str.find_first_of("."));
		//closed or open?
		//printf("%s\n", rstr.c_str());
		string fstr;
		if(rstr.find_first_of(".") == string::npos)
		    fstr = rstr+".f";
		else
		    fstr = rstr+"f";

		if (str.find_first_of("-_") != string::npos) {
			//this means closed
			string str1 = str.substr(0, str.find_first_of("-_"));
			string str2 = str.substr(str.find_first_of("-_")+1);
			int s = atoi(str1.c_str());
			int i = atoi(str2.c_str());
			read_feat(feat[readcount], fstr);
			featclasses[readcount] = s;
			featindices[readcount] = i;

		}
		else {
			//this means open
			read_feat(feat[readcount], fstr);
			featindices[readcount] = atoi(str.c_str());
			featclasses[readcount] = -1;
		}
		readcount += 1;
	}
}

int read_filelen(string fname) {
	int featlen = 0;

	//one round to learn the length...
	ifstream fread;
	fread.open(fname.c_str());
	while (fread.peek() != EOF) {
		string str = "";
		getline(fread, str);
		featlen += 1;
	}
	fread.close();

	return featlen;
}

void read_options(string fname) {
//	std::map <string, string> d;
	ifstream fread;
	fread.open(fname.c_str());
	while (fread.peek() != EOF) {
		string str = "";
		getline(fread, str);
		while (str.find("#") != string::npos)
			str = str.substr(0, str.find_first_of("#"));
		std::replace( str.begin(), str.end(), '\t', ' ');
		if (str.find(" ") != string::npos) {
			string optname = str.substr(0, str.find_first_of(" "));
			string optval = str.substr(str.find_last_of(" ")+1);
			d[optname] = optval;
		}
	}
	fread.close();
}

int get_featnum(string folder) {
	//Guess feat num so feat set can be changed without changing this code

	ostringstream freadnamestream;
	freadnamestream << folder << d["FIRST_FILE"];
	string freadname = freadnamestream.str();

	ifstream fread;
	fread.open(freadname.c_str());
	string str = "";
	getline(fread, str);
	fread.close();

	int feat_count = 0;
	for (int i = 0; i < str.length(); i++) {
		if (str[i] == ' ') {
			feat_count += 1;
		}
	}

    printf("Getting num features from file %s%s: is %d\n", folder.c_str(), d["FIRST_FILE"].c_str(), feat_count);

	return feat_count;
}


int main(int argc, char** argv) {
	/*int OPENTEST_list [6] = {100, 500, 1000, 3000, 5000, 6000};
	int NEIGHBOUR_list [10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	if(argc == 3){
		int OPENTEST_ind = atoi(argv[1]);
		int NEIGHBOUR_ind = atoi(argv[2]);

		OPEN_INSTNUM = OPENTEST_list[OPENTEST_ind % 5];
		NEIGHBOR_NUM = NEIGHBOUR_list[NEIGHBOUR_ind % 10];
	}*/

	srand(time(NULL));

	if(argc != 2){
	    cout <<"call: ./flearner optname"<<endl;
	    exit(1);
	}
	char* optionname = argv[1];
	read_options(string(optionname));

	open_world = (atoi(d["OPEN_INSTNUM"].c_str()) > 0);

	if (d.find("NEIGHBOR_NUM") == d.end()) {
	    // default number of neighbors
	    if (open_world)
            NEIGHBOR_NUM = 2;
	    else
            NEIGHBOR_NUM = 1;
    }
    else
        NEIGHBOR_NUM = atoi(d["NEIGHBOR_NUM"].c_str());


	if (d.find("FEAT_NUM") != d.end()) {
	    FEAT_NUM = atoi(d["FEAT_NUM"].c_str());
	}
	else
	    FEAT_NUM = get_featnum(d["CELLF_LOC"]);

	printf("num features %d, open_world=%d, num_neighbors=%d, %s\n", FEAT_NUM, open_world, NEIGHBOR_NUM, d["CELLF_LOC"].c_str());

	//learn weights
//	int wlen = read_filelen(d["TRAIN_LIST"]);
//	if(wlen == 0) {
//	    printf("No weights have been given in %s, exit\n", d["TRAIN_LIST"].c_str());
//	    exit(1);
//	}
//	float** wfeat = new float*[wlen];
//	int * wfeatclasses;
//	int * wfeatindices;
//	for (int i = 0; i < wlen; i++) {
//		wfeat[i] = new float[FEAT_NUM];
//	}
//	wfeatclasses = new int[wlen];
//	wfeatindices = new int[wlen];
//	read_filelist(wfeat, wfeatclasses, wfeatindices, wlen, d["TRAIN_LIST"]);

	float** trainfeat;
	int * trainfeatclasses;
	int * trainfeatindices;
	int trainlen = read_filelen(d["TRAIN_LIST"]);
	trainfeat = new float*[trainlen];
	for (int i = 0; i < trainlen; i++) {
		trainfeat[i] = new float[FEAT_NUM];
	}
	trainfeatclasses = new int[trainlen];
	trainfeatindices = new int[trainlen];
	read_filelist(trainfeat, trainfeatclasses, trainfeatindices, trainlen, d["TRAIN_LIST"]);

	float * weight = new float[FEAT_NUM];
	alg_init_weight(weight);
	clock_t t1, t2;
	float train_time, test_time;
	t1 = clock();
	alg_recommend2(trainfeat, trainfeatclasses, trainlen, weight);
	float tot_weight = 0;
	for(int i=0;i<FEAT_NUM;i++) {
	    if (isnan(weight[i])) cout << weight[i] << endl;
	    tot_weight += weight[i];
	}
	cout << "Total weight is " << tot_weight << endl;
	t2 = clock();
	train_time = (t2 - t1)/float(CLOCKS_PER_SEC);


	//load training instances
//	float** trainfeat;
//	int * trainfeatclasses;
//	int * trainfeatindices;
//	int trainlen = read_filelen(d["TRAIN_LIST"]);
//	trainfeat = new float*[trainlen];
//	for (int i = 0; i < trainlen; i++) {
//		trainfeat[i] = new float[FEAT_NUM];
//	}
//	trainfeatclasses = new int[trainlen];
//	trainfeatindices = new int[trainlen];
//	read_filelist(trainfeat, trainfeatclasses, trainfeatindices, trainlen, d["TRAIN_LIST"]);

	//Load testing instances
	float** testfeat;
	int * testfeatclasses;
	int * testfeatindices;
	int testlen = read_filelen(d["TEST_LIST"]);
	testfeat = new float*[testlen];
	for (int i = 0; i < testlen; i++) {
		testfeat[i] = new float[FEAT_NUM];
	}
	testfeatclasses = new int[testlen];
	testfeatindices = new int[testlen];
	read_filelist(testfeat, testfeatclasses, testfeatindices, testlen, d["TEST_LIST"]);
	printf("Training and testing instances loaded\n");

	int tpc, tnc, pc, nc;
	t1 = clock();
	accuracy(trainfeat, testfeat, trainfeatclasses, testfeatclasses, testfeatindices, trainlen, testlen, weight);
	t2 = clock();
	test_time = (t2 - t1)/float(CLOCKS_PER_SEC);

//	for (int i = 0; i < wlen; i++) {
//		delete[] wfeat[i];
//	}
//	delete[] wfeat;
//	for (int i = 0; i < trainlen; i++) {
//		delete[] trainfeat[i];
//	}
//	delete[] trainfeat;
//	for (int i = 0; i < testlen; i++) {
//		delete[] testfeat[i];
//	}
//	delete[] testfeat;
//
//	delete[] weight;
	return 0;
}
