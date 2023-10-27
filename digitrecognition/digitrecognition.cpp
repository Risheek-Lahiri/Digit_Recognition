// 234101042_digitrecognition.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#pragma warning (disable : 4996)  //to disable fopen() warnings
#include <stdio.h>
#include <stdlib.h>		//atoi, atof
#include <string.h>		//strcpy
#include <conio.h>		//getch,
#include <time.h>		// time(NULL)
#define _USE_MATH_DEFINES	// for pi
#include <math.h>       // abs, log, sin, floor
#include <windows.h>
#pragma comment(lib,"Winmm.lib")	// lib for listenning to sound


#define p 12						 //	Number of Cepstral Coefficients, dimensionality of vector x
//#define codeBkSize 32		//change // CodeBook Y Size = K = M		
//#define distr_delta 0.000001			 // 0.00001 or 0.000001 abs(old_distortion - new_dist) > delta. 
#define W 10				//change // Number of Words/Digits//HMM
#define N 5					//change // Number of States per HMM
#define M 32				//change // Number of Distinct Observation Symbols
#define Tmax 150			//change // Max Length of Observation Sequence
#define On 20				//change // Number of Training Observations Given in one file
#define Ot 20				//change // Number of Testing Observations Given in one file
#define model_iterations 200				//change // Number of times Re-estimation has to be done
#define repeatConvergence 2

#define TRAINING 1
#define TESTING 2
#define InputFolderModel 1
#define OutputFolderModel 2

//debug
bool segregate_speech = true;			// True: to segreagate speech data with respect to start and end marker in its output folder. 
bool segregate_Live_speech = true;
bool showCoefficientsInFile = false;		// True: Show Coefficients Values R, A, C's in each voice analysised files
bool showAlphaBetaPstarInConsole = false;	//True: to show alpa, beta probabilities in the console for each observation sequence
bool showStateSeqAlphaBetaInFileForEachObsAfterConverge = true;



// HMM Lamda variable declaration
long double PI[N];			// Initial State Distribution
long double A[N][N];		// State Transition Probability Distribution
long double B[N][M];		// Observation Symbol Probability Distribution						// 
int O[On>Ot?On:Ot][Tmax];			// N observation Sequence Length of T (for training, testing), Observation Sequence Given, Codebook Index Entry
int OFmax[On>Ot?On:Ot];				// max frames in each observation sequence

int offline_correct_count=0;	// to detect number of utterance of digit correctly recognized
int offline_overall_count=0;	// to detect number of utterance of digit correctly recognized in all files

// P1 Evaluation Problem (Scoring Problem)
long double alpha[Tmax][N]; // T*N, forward procedure variable, Prob of Partial Observation Seq O1 to Ot until time t and state Si  at time t given model lambda
long double beta[Tmax][N];  // T*N, backward procedure variable, Prob of Partial Obser Seq O(t+1) to Ot given state Si at time t and mdodel lamda
long double probability_alpha, probability_beta;

// P2 Uncovering the Problem
long double gamma[Tmax][N]; // T*N, Prob of being in state Si at time t given the Obser Seq and model lamda
long double delta[Tmax][N]; // T*N, Viterbi Algo variable, best score along a single path, at time t which accounts for the first t obser and ends in state Si
int psi[Tmax][N];   // T, Viterbi Algo variable, Book Keeping Var to keep track of argument that maximized the delta
long double Pstar, Pstar_old;	 // Viterbi Algo variable, max probability of delta_T_i 
int Qstar[Tmax];	 // Viterbi Algo variable, State Sequence path

// P3 Reestimation Problem
long double XI[Tmax][N][N];		// Xai Matrix
long double PI_BAR[N];			// Re-estimated Initial State Distribution
long double A_BAR[N][N];		// Re-estimated State Transition Probability Distribution
long double B_BAR[N][M];		// Re-estimated Observation Symbol Probability Distribution
long double converged_A[On][N][N];		// save all the converged A matrix of training Sequence
long double converged_B[On][N][M];		// save all the converged B matrix of training sequence			
long double A_Prev[N][N];		// Previous Averaged A Matrix
long double B_Prev[N][M];		// Previous Averaged B Matrix

// CodeBook
long double codebook[M][p];
const char codebook_file_name[] = "codebook.txt";
bool codebook_universe_generation = false;
// Files								
const char input_folder[] = "input_lamda/";
const char output_folder[] = "output/";
const char WordFolderName[] = "Digit";
const char *LambdaFileNames[] = {"A_","B_","Pi_","obs_seq_training_", "obs_seq_testing_"};
const char *WordNames[]={"0","1","2","3","4","5","6","7","8","9"};
const char voice_data_prefix[] = "234101042_";
const char output_folder_Model_name[] = "Models";

FILE *fp_console ;		//to write output file		
char completePathOuput[200];

//Live Voice
time_t timestamp;				//timestamp for live voice filename
char liveRecordingCommand[300], liveRecordingFileName[100];
const char recording_module_exe_path[] = "RecordingModule\\Recording_Module.exe";
const char input_live_voice_folder[] = "input_live_voice_data/";
#define liveRecordingWAV "input_live_voice_data/live_input.wav"


// Observation Sequence Generations Functions	

//Speech Samples
#define scaleAmp 5000							  // Max Amplitutde Value to Scale.
#define initIgnoreHeaderLines 5			//change  // Number of Initial Header Lines in Txt Files to Ignore.
#define initIgnoreSamples 6400				//change(30frames)  // Number of Initial samples to ignore.
#define initNoiseFrames 10				//change  // Number of Initial Noise Frames to Consider.
#define thresholdNoiseToEnergyFactor 3	//change  // Noise to Energy Factor.
#define samplingRate 16000						  // Sampling Rate of Recording.
#define sizeFrame 320		//change // Number of Samples per Frame.
#define initIgnoreEndFrames 30

const unsigned short totDigits = W;
const unsigned short MaxFrames = Tmax+10;				//change	// max number of frames to consider overall.
const unsigned short q=p;				//change	//#Coefficients (c_i's) that need to be found
const unsigned short NumOfTrainingFiles = On;
const unsigned short NumOfTestingFiles = Ot;

// DISTANCE, WEIGHTS
const double w_tkh[]={1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};		//tokhura weigths provided.

// COEFFICIENTS
double w_rsw[q+1];						// weights for raised sine window.
double C_rsw[Tmax][q+1]; // calculating Raised Sine Window Cepstral coefficient and storing for each frame
int OBS_SEQ[Tmax];

// Speech
int samples[500000], maxAmp = 0;							//max number of samples in recording to consider, maximum amplitute for rescaling.
double normSamples[500000];									//normalised samples.
long cntTotSamples = 0, cntTotFrames = 0;					// Total Samples in recording, Total Frames in Recording.
float DCshift =0;
long start=10, end=Tmax-10 ;										//start and end marker, Frames

// ZCR and ENERGY
float cntZCR[MaxFrames], avgZCR[MaxFrames], avgEnergy[MaxFrames];  
float totEnergy=0, noiseEnergy=0, initNoiseZCR=0, initNoiseAvgZCR=0;;
float thresholdZCR=0;
double thresholdEnergy=0;
double maxEnergy = 0;										// Max Energy of the Frame.
int STE_Marker;												// Max STE Marker For the Frame

// File Stream
FILE *fp_obs_seq_ip, *fp_obs_seq_norm, *fp_obsseq_norm_seg, *fp_obsseq_final_op, *fp_obsseq_console;							// file pointer for input stream and output stream.
char OSeqfileNameIp[300], OSeqcompletePathIp[300], OSeqcompletePathNorm[300], OSeqcompletePathNormSeg[300], OSeqcompletePathFinOp[300], OSeqcompletePathConsole[300];
char OSeqcharsLineToRead[50];															// number of characters to read per line in file.
const char filePathInputVoiceTraining[] = "New folder\input_voice_testing_data/";		 							// Folder name where all Vowels recordings are placed 1 to 100.	input_digits_data/
const char filePathInputVoiceTesting[] = "New folder\input_voice_testing_data/";	
const char fileOutputRecordingNorSeg[] = "output_voice_recordings_normalised_segregated/";							// Folder name where output of segregated digits recordings are placed and its normalised files. output_recordings/
const char fileOutputRecordingAnalysis[] = "output_voice_recordings_analysis_files/";								// Folder name where test file analysis are saved. output_test_result/


//Calculating w[m] weights for raised sine window and storing in array for future use.

void CalculateWeightsForRaisedSineWindow(){
	for(int i=1;i<=q;i++){
		w_rsw[i] = 1 + (q/2)*sin(M_PI*i/q);
	}
}


//Calculating R[i] Values using Auto Correlation Method
//Input:	   *s is the pointer to our samples array.
//Output:	   *Rn is the pointer to our R array for storing final Ri values (0 to p).

void calculateRi_AutoCorrelation(double *s, double *Rn){
	for(int k=0;k<=p;k++) 
	{ 
		Rn[k]=0;
		for(int m=0; m < sizeFrame-k; m++)			
		{
			Rn[k] += s[m]*s[m+k];				// R[0] R[1] .... R[p]
		}	
	}
}

//Calculating A[i] Values using Levinson Durbin Process
//Input:	   *R is the pointer to our Auto Correlated Energy part.
//Output:	   *A is the pointer to our A array for storing final Ai values (1 to p).

void calculateAi_Durbin(double *R, double *A){
	 double E[p+1] = {0};
	 double k[p+1] = {0};
	 double alpha[p+1][p+1] = {0};
	 double sum=0;
	 int i, j;

	E[0]=R[0]; //Initialize E[0] with energy

	for( i=1;i<=p;i++)
	{
		if(i==1)
			k[1]=R[1]/R[0]; //special case i=1
		else //find k(i) for all other remaining values of i
		{
			 sum=0;
			for( j=1;j<=i-1;j++)
			{
				sum+=alpha[i-1][j]*R[i-j];
			}
			k[i]=(R[i]-sum)/E[i-1];
		}
		alpha[i][i]=k[i];

		for( j=1;j<=i-1;j++)
			alpha[i][j]=alpha[i-1][j]-k[i]*alpha[i-1][i-j]; //update coefficients from previous values
		
		E[i]=(1-k[i]*k[i])*E[i-1]; //update E[i]
	}
	for( i=1;i<=p;i++){	
		//A[i] = -1*alpha[p][i];
		A[i] = alpha[p][i];				// A[0] A[1] .... A[p]
	}
}


//Calculating C[i] Cepstral Coefficient Values.
//Input:	   *A is the pointer to our Ai Array.
				//sigma is R[0] value.
//Output:	   *A is the pointer to our A array for storing final Ai values (1 to p).

void calculateCi_Cepstral(double sigma, double *A, double *c){
	int k,m;
	double sum=0;

	c[0]= logl(sigma*sigma); 
	for(m=1; m<=p; m++)
	{
		sum=0;
		for(k=1;k<=m-1;k++) //sum from older cepstral coefficents to compute new cepstral coefficients
			sum+=k*c[k]*A[m-k]/m;
		c[m]=A[m]+sum;		//new cepstral coefficients
	}
	/*
		if(m>p) // For This Assignment this never get executed as we assume q=p
		{
			for(;m<=q;m++)
			{
				sum=0;
				for(k=1;k<=m-1;k++) //sum from older cepstral coefficents to compute new cepstral coefficients
					sum+=k*c[k]*A[m-k]/m;
				c[m]=sum;		//new cepstral coefficients
			}
		}
	*/
}



	//To Display Common Settings used in our System
	//Input: File Pointer in case things needed to be written on file.

void DisplayCommonSettingsObsSeq(FILE *fp_op=NULL){
	// General Information to Display
	if(fp_op==NULL){
		printf("****-------- --------****\n");		
		printf("-Common Settings are : -\n");	
		printf(" P (=Q) = : %d\n", p);
		printf(" Frame Size : %d\n", sizeFrame);	
		printf(" Num of Files For Training : %d\n", NumOfTrainingFiles);	
		printf(" Num of Files For Testing : %d\n", NumOfTrainingFiles);	

		printf(" Tokhura Weights : ");
		for(int i=0; i<q; i++){
			printf("%0.1f(%d) ", w_tkh[i],i+1);
		}
		printf("\n");
		printf(" Amplitutde Value to Scale : %d\n", scaleAmp);			
		printf(" Intital Header Lines Ignore Count : %d\n", initIgnoreHeaderLines); 
		printf(" Intital Samples to Ignore : %d\n",initIgnoreSamples);	
		printf(" Intital Noise Frames Count : %d\n",initNoiseFrames);	
		printf(" Noise to Energy Factor : %d\n",thresholdNoiseToEnergyFactor); 
		printf(" Sampling Rate of Recording: %d\n",samplingRate); 
		printf("----------------------------------------------------------------\n\n");		
	}
	else if(fp_op!=NULL){
		fprintf(fp_op, "****----------------****\n");		
		fprintf(fp_op, "-Common Settings are : -\n");	
		fprintf(fp_op, " P (=Q) = : %d\n", p);
		fprintf(fp_op, " Frame Size : %d\n", sizeFrame);	
		fprintf(fp_op, " Num of Files For Training : %d\n", NumOfTrainingFiles);	
		fprintf(fp_op, " Num of Files For Testing : %d\n", NumOfTrainingFiles);	

		fprintf(fp_op, " Tokhura Weights : ");
		for(int i=0; i<q; i++){
			fprintf(fp_op, "%0.1f(%d) ", w_tkh[i],i+1);
		}
		fprintf(fp_op, "\n");
		fprintf(fp_op, " Amplitutde Value to Scale : %d\n", scaleAmp);			
		fprintf(fp_op, " Intital Header Lines Ignore Count : %d\n", initIgnoreHeaderLines); 
		fprintf(fp_op, " Intital Samples to Ignore : %d\n",initIgnoreSamples);	
		fprintf(fp_op, " Intital Noise Frames Count : %d\n",initNoiseFrames);	
		fprintf(fp_op, " Noise to Energy Factor : %d\n",thresholdNoiseToEnergyFactor); 
		fprintf(fp_op, " Sampling Rate of Recording: %d\n",samplingRate); 
		fprintf(fp_op, "----------------------------------------------------------------\n\n");	
	}
}


	//Normalising and DC Shift of Samples.
	//Input (global):		fp_obs_seq_ip is to read from recording text file.
						//fp_obs_seq_norm is to save normalised samples into another file.
						//fp_obsseq_console is to save analysed values in common output file.

void normalize_dcshift_samples(){
			cntTotSamples=0; maxAmp = 0;

			int totIgnore;
				 totIgnore=initIgnoreHeaderLines + initIgnoreSamples; //5 + 2 = 7
			// totIgnore=0;
			long sample_index = totIgnore+1; // till 7 samples ignored, sample count is 8, so to make array index 0 there is +1 
			long sample_index_norm = 0;
			double normFactor = 0;
			double normOutput = 0;

				while(!feof(fp_obs_seq_ip)){
					fgets(OSeqcharsLineToRead, 50, fp_obs_seq_ip);
					cntTotSamples += 1 ;  

					if(cntTotSamples > totIgnore){
						sample_index_norm = cntTotSamples - sample_index;
						samples[sample_index_norm] = (int)atoi(OSeqcharsLineToRead);
						DCshift += samples[sample_index_norm];
								if(abs(samples[sample_index_norm]) > maxAmp)
									maxAmp = abs(samples[sample_index_norm]);
					}
				}

			cntTotSamples = cntTotSamples - totIgnore;		// total number of samples stored in array
			DCshift = DCshift/cntTotSamples;				//average DC Shift
			cntTotFrames = (int)floor((1.0*cntTotSamples)/sizeFrame);			//total number of frames.

			start=10;
			end = cntTotFrames-10;

			normFactor = (double)scaleAmp/maxAmp;			//normalising factor

			// saving in normalised file
			for(long i=0; i<cntTotSamples; i++){
				normOutput = (double)(samples[i] - DCshift)*normFactor;
				normSamples[i]=normOutput;
				fprintf(fp_obs_seq_norm, "%lf\n", normSamples[i]);
			}
			
			
	//printf(" TOTAL SAMPLES : %d\n TOTAL FRAMES : %d\n DC SHIFT needed : %lf\n Maximum Amplitude : %d\n Normalization Factor : %lf\n ", cntTotSamples, cntTotFrames, DCshift, maxAmp, normFactor);
	if(fp_obsseq_console!=NULL){
		fprintf(fp_obsseq_console, " TOTAL SAMPLES : %d\n TOTAL FRAMES : %d\n DC SHIFT needed : %lf\n Maximum Amplitude : %d\n Normalization Factor : %lf\n ", cntTotSamples, cntTotFrames, DCshift, maxAmp, normFactor);
	}
}


	//Calculating ZCR and Energy of Frames
	//Input (global): fp_obs_seq_norm is to read from normalised file of the samples.

void zcr_energy_frames(){
	rewind(fp_obs_seq_norm);

	long i,j;
	float s_i, s_i_1=1;

	//totEnergy=0;
	 maxEnergy = 0;										// Max Energy of the Frame.
	 STE_Marker = 0;	
	
	for(i=0;i < cntTotFrames;i++)
		{
			cntZCR[i]=0;
			avgEnergy[i]=0;
			for(j=0;j < sizeFrame ;j++)
			{
				fgets(OSeqcharsLineToRead, 50, fp_obs_seq_norm); // reading from normalised input
				s_i = (float)atof(OSeqcharsLineToRead);
				avgEnergy[i] += (s_i*s_i);
				cntZCR[i] +=  (s_i_1*s_i < 0);
				s_i_1 = s_i;
			}	
			avgEnergy[i]/=sizeFrame;
			avgZCR[i] = cntZCR[i]/sizeFrame;
			//totEnergy+=avgEnergy[i];
			//fprintf(fp_obs_seq_norm, "%f %0.1f \n", avgEnergy[i], cntZCR[i]);	//dumping the features of frames.
			// calculation for detecting STE Frame
			if(avgEnergy[i] > maxEnergy){
				maxEnergy = avgEnergy[i];
				STE_Marker=i;
			}
		}

}



	//Calculating ZCR and Energy of Noise Frames, and Finally Thresholds for ZCR and Energy
	//Input (global): fp_obsseq_console is to save analysed values in common output file.

void noiseEnergy_thresholds_frames(){
	noiseEnergy=0; initNoiseZCR=0; initNoiseAvgZCR=0;
	int i;
	for(i=0;i < initNoiseFrames;i++){
			initNoiseZCR+=cntZCR[i];
			initNoiseAvgZCR+=avgZCR[i];
			noiseEnergy+=avgEnergy[i];
	}
		thresholdZCR=initNoiseZCR/initNoiseFrames;
		noiseEnergy/=initNoiseFrames;
		thresholdEnergy=noiseEnergy*thresholdNoiseToEnergyFactor;

	//printf( "\n---- Initial Noise Frames : %d ----\n\n", initNoiseFrames);
	//printf(" Avg Noise Energy : %lf\n Total Noise ZCR : %0.1f\n Threshold ZCR : %0.1f\n Threshold Energy(Avg Noise*%d) : %0.5lf\n ", noiseEnergy, initNoiseZCR, thresholdZCR, thresholdNoiseToEnergyFactor, thresholdEnergy);
	if(fp_obsseq_console!=NULL){
		fprintf(fp_obsseq_console, "\n---- Initial Noise Frames : %d ----\n\n", initNoiseFrames);
		fprintf(fp_obsseq_console, " Avg Noise Energy : %lf\n Total Noise ZCR : %0.1f\n Threshold ZCR : %0.1f\n Threshold Energy(Avg Noise*%d) : %0.5lf\n ", noiseEnergy, initNoiseZCR, thresholdZCR, thresholdNoiseToEnergyFactor, thresholdEnergy);
	}
	
}



	//Detecting Start and End Marker of the Frame.
	//Input (global): fp_obsseq_console is to save analysed values in common output file.
					//fp_norm_Seg is to save normalised segregated samples into another file.

void marker_start_end_segregated(){
	bool flag=false;		//to detect start mark
	// -3 to ignore last 3 frames.
	for(int i=0; i<cntTotFrames-initIgnoreEndFrames; ++i){
			if(!flag && avgEnergy[i+1] > thresholdEnergy && avgEnergy[i+2] > thresholdEnergy && avgEnergy[i+3] > thresholdEnergy && avgEnergy[i+4] > thresholdEnergy){ //&& avgEnergy[i+4] > thresholdEnergy && avgEnergy[i+5] > thresholdEnergy
					start = i;
					flag = 1;
			}
			else if(flag && avgEnergy[i+1] <= thresholdEnergy && avgEnergy[i+2] <= thresholdEnergy && avgEnergy[i+3] <= thresholdEnergy && avgEnergy[i+4] <= thresholdEnergy ){ //&& avgEnergy[i+4] < thresholdEnergy && avgEnergy[i+5] < thresholdEnergy
				end = i;
				flag = 0;
				break;
			}
		}
	
	if(flag == 1) end = cntTotFrames - initIgnoreEndFrames; //if end is not found then making the last frame - 3 as the end marker for the word

	long startSample= (start+1)*sizeFrame;
	long endSample= (end+1)*sizeFrame;
	long totFramesVoice = end-start+1;
	
//saving segregated voice data in different file
	if(fp_obsseq_norm_seg!=NULL && (segregate_speech || segregate_Live_speech)){
		for(long i=startSample; i<endSample; i++){
			fprintf(fp_obsseq_norm_seg, "%lf\n", normSamples[i]);
		}
	}
	

		//printf("\n---- Segregated Data Saved in File: %s ----\n\n", OSeqcompletePathNormSeg);
		//printf(" START Frame : %ld\t END Frame : %ld\t Total Frames : %ld\n", start+1, end+1, totFramesVoice);
		//printf(" Starting Sample : %ld\t Ending Sample : %ld\n", startSample, endSample);
		//printf("\n--------\n");
		//printf("\n------------------------------------------------------------------------\n");

	double starting_time = (1.0*startSample)/(samplingRate);
	double ending_time = (1.0*endSample)/(samplingRate);
	if(fp_obsseq_console!=NULL){
		fprintf(fp_obsseq_console, "\n---- Segregated Data Saved in File: %s ----\n\n", OSeqcompletePathNormSeg);
		fprintf(fp_obsseq_console, " START Frame : %ld\t END Frame : %ld\t Total Frames : %ld\n", start+1, end+1, totFramesVoice);
		fprintf(fp_obsseq_console, " Starting Sample : %ld\t Ending Sample : %ld\n", startSample, endSample);
		fprintf(fp_obsseq_console, " Starting Time (seconds) : %lf\t Ending Time (seconds) : %lf\n", starting_time, ending_time);
		fprintf(fp_obsseq_console, " Max STE Marker Frame : %d\n", STE_Marker);	
		fprintf(fp_obsseq_console, "\n--------\n");
		fprintf(fp_obsseq_console, "\n------------------------------------------------------------------------\n");
	}
}



	//Reading CodeBook From the File.
void read_codebook_from_file(){
	FILE *fp_cb ;			//to read codebook
	char completePathCB[200];
	sprintf(completePathCB, "%s%s", input_folder, codebook_file_name);  // {input_lamda/}+{codebook_file_name}

	fp_cb = fopen(completePathCB, "r");				//to read input codebook
	if(fp_cb == NULL){ 
				perror("\n Error: ");
				printf("\n codebook File Name is: %s", completePathCB);
				getch();
				return;
	}

	//printf("\n ----------------------- Reading CODEBOOK from File: %s ----------------------- \n", completePathCB);	
		//fprintf(fp_console, "\n ----------------------- Reading CODEBOOK from File: %s ---------------------- \n", completePathCB);	

	for (int i = 0; i < M; ++i){
		//printf("M[%d]\t", i+1);
			//fprintf(fp_console, "M[%d]\t", i+1);
		for (int j = 0; j < p; ++j){
				fscanf(fp_cb,"%Lf",&codebook[i][j]);
				//printf("%Lf (%d)\t",codebook[i][j], j+1);
				//fprintf(fp_console, "%Lf (%d)\t",codebook[i][j], j+1);
			}
		//printf("\n");
		//fprintf(fp_console, "\n");
	}

	fflush(fp_cb); fclose(fp_cb); 
}//read_codebook_from_file

	//Calculating Coefficients Ai's, Ci's for each frame of the segregated voice data.

void calculateCoefficientsForFramesOfSpeech(const long totFramesVoice){

	double markedSamples[sizeFrame]={0};

	//long startSample= (start+1)*sizeFrame;
	//long endSample= (end+1)*sizeFrame;
	//long totFramesVoice = end-start+1;

	long int startOfMarkedFrame = (start+1);		

	//int skipCounter=0;
	for(int ff=0; ff<totFramesVoice; ff++){

		long SampleMarkerToCopy= (startOfMarkedFrame + ff)*sizeFrame;	

		for(int i=0; i<sizeFrame; i++){
			markedSamples[i] = normSamples[i + SampleMarkerToCopy];
		} 


	// Calculating R_i values using AutoCorrelation Method.
		double R[p+1] = {0};
			calculateRi_AutoCorrelation(markedSamples, R);
			///* Not Applicable for this assignment as we are taking Max STE Frame.
			if(R[0]==0){
				printf("\n R[0] Energy should not be ZERO, Skipping frame %d, \n", (startOfMarkedFrame + ff));
				system("pause");
				//skipCounter++;
				continue;
			}
			
		// calculating A_i using Durbin algo.
		double A[p+1] = {0};
			calculateAi_Durbin(R, A);

		// calculating Cepstral coefficient.
		double C[q+1] = {0};
			calculateCi_Cepstral(R[0], A, C);

		if(fp_obsseq_console!=NULL && showCoefficientsInFile)
		{
			fprintf(fp_obsseq_console, "\n\t\t\tFile Name: %s, Frame: %d", OSeqfileNameIp, startOfMarkedFrame +ff);
			fprintf(fp_obsseq_console, "\n\nR Coefficient Values \t LPC Coefficient values \t Cepstral Coefficient values \t Raised Sine Window\n");
			fprintf(fp_obsseq_console, "R[%d] = %lf \n",0, R[0]);
		}
		// Final Values of coefficients
		
		for(int i=1;i<=p;i++){	
				C_rsw[ff][i] = C[i]*w_rsw[i];   
			if(fp_obsseq_console!=NULL && showCoefficientsInFile){	
				fprintf(fp_obsseq_console, "R[%d] = %lf \t ",i, R[i]);
				fprintf(fp_obsseq_console, "A[%d] = %lf \t ", i, A[i]);
				fprintf(fp_obsseq_console, "C[%d] = %lf \t", i, C[i]);	
				fprintf(fp_obsseq_console, "*(%lf)=> C[%d] = %lf \n", w_rsw[i], i, C_rsw[ff][i]);
			}
		}
		/* For This Assignment this never get executed as we assume q=p
		for( i=p+1;i<=q;i++){
			C_rsw[ff][i] = C[i]*w_rsw[i];  
		}
		*/
		
	}//end of ff loop
}//calculateCoefficientsForFramesOfSpeech


	//Generate Observation Sequence Based on the Ci's of the frame and Minimum Tokhura Distance

void generateObeservationSequence(const long totFramesVoice)
{
	
	double temp;
	double td_cb[M] = {0};  //save min distance of Frame Ci's with each Codebook Index;
	int min_index=0;

	for(int ff=0; ff<totFramesVoice; ff++){

		for (int m= 0; m < M; ++m){
				td_cb[m]=0;
			for (int k = 1; k <= p; ++k){
				temp = C_rsw[ff][k] - codebook[m][k-1];
				td_cb[m] += w_tkh[k-1]*temp*temp;
			}// for each k col of codebook
		}// for each m row index of codebook

		min_index=0;
		for (int m= 0; m < M; ++m){
			if(td_cb[m] < td_cb[min_index]){
				min_index = m;
			}
		}// for each m row index of codebook

		OBS_SEQ[ff] = min_index+1;

	}//end of ff loop

}//generateObeservationSequence


	//Print Observation Sequence in the file
void print_observation_sequence(const long totFramesVoice, const int digit){

	fprintf(fp_obsseq_final_op, "-------------------------------- %d:%d -------------------------\n", digit, totFramesVoice);
	for(int ff=0; ff<totFramesVoice; ff++){
		fprintf(fp_obsseq_final_op, "%d ", OBS_SEQ[ff]);
	}
	fprintf(fp_obsseq_final_op, "\n");

}//print_observation_sequence


	//Generate Observation Sequence For Recordings present in the disk

int sequence_generation(unsigned short int seq_type)
{

	unsigned short NumOfFiles;
	char OSeqfilenameIpformatSpecifier[50];
	char filePathInputVoice[50];
	char lambda_obs_seq_file_name[50];
	char seq_type_name[20];
	char obs_seq_name[50]="OBSERVATION SEQUENCE FOR";

	if(seq_type == 1)
	{	
		NumOfFiles = On;
		strcpy(filePathInputVoice, filePathInputVoiceTraining);
		strcpy(OSeqfilenameIpformatSpecifier, "training_%s_%s_%s%d");
		strcpy(lambda_obs_seq_file_name, LambdaFileNames[3]);
		strcpy(seq_type_name, "TRAINING");
	}
	else if(seq_type == 2)
	{	
		NumOfFiles = Ot;
		strcpy(filePathInputVoice, filePathInputVoiceTesting);
		strcpy(OSeqfilenameIpformatSpecifier, "testing_%s_%s_%s%d");
		strcpy(lambda_obs_seq_file_name, LambdaFileNames[4]);
		strcpy(seq_type_name, "TESTING");
	}

	if(codebook_universe_generation)
	{
		//to save Cepstral Coefficients 
		sprintf(OSeqcompletePathFinOp, "%sUniverse.csv", input_folder);  		
		fp_obsseq_final_op = fopen(OSeqcompletePathFinOp, "w"); 
		if(fp_obsseq_final_op==NULL){ 
			perror("\n Error: ");
			printf("\n File Name : \n  %s\n", OSeqcompletePathFinOp);
			getch();
			return 1;
		}
		strcpy(obs_seq_name, "Cepstral Coefficients of Frames in");
	}

	for(int d = 0 ; d<totDigits ; d++) //iterating through all digits. totDigits
	{			
		
		//if(codebook_universe_generation)
		//{
			printf("\n\n\tGENERATING %s (%s): %s %s :-\n",obs_seq_name, seq_type_name, WordFolderName,  WordNames[d]);
		//}
		//else
		//{
		//	printf("\n\t ---#---#---#---#---#--- GENERATING OBSERVATION SEQUENCE FOR (%s): %s %s ---#---#---#---#---#---\n", seq_type_name, WordFolderName,  WordNames[d]);
		//}
		for(int fileCounter=1; fileCounter <= NumOfFiles ; ++fileCounter)//iterating through all files of given digits (1 to X).
		{
		// Creating necessary file Path for data
			
			//input file name
			sprintf(OSeqcompletePathIp, "%s%s/%s%d.txt", filePathInputVoice, WordNames[d], voice_data_prefix, fileCounter); // filePathInputVoiceTraining/ {0} + / + {obs_} + {1}.txt
			//segregated file data name
			sprintf(OSeqfileNameIp, OSeqfilenameIpformatSpecifier, WordFolderName, WordNames[d], voice_data_prefix, fileCounter); //OSeqfileNameIp = {Digit} +_+ {0} +_+ {obs_} + {1}
			sprintf(OSeqcompletePathNorm, "%s%s_normalized_samples.txt", fileOutputRecordingNorSeg, OSeqfileNameIp); // fileOutputRecordingNorSeg/ {OSeqfileNameIp}_normalized_samples.txt
			sprintf(OSeqcompletePathNormSeg, "%s%s_normalized_segregated_data.txt", fileOutputRecordingNorSeg, OSeqfileNameIp); // fileOutputRecordingNorSeg/ {OSeqfileNameIp} + _normalized_segregated_data.txt
			//to save analysis file
			sprintf(OSeqcompletePathConsole, "%s%s_analysis.txt", fileOutputRecordingAnalysis, OSeqfileNameIp);  // {fileOutputRecordingAnalysis/}+{OSeqfileNameIp}+"_analysis.txt 
			/**************** Opening respective files. ****************/
			fp_obs_seq_ip = fopen(OSeqcompletePathIp, "r");				//to read input file
			fp_obs_seq_norm = fopen(OSeqcompletePathNorm, "w+");		//to save normalised samples
			fp_obsseq_norm_seg = fopen(OSeqcompletePathNormSeg, "w");  //to save segregated recording from start to end
			fp_obsseq_console = fopen(OSeqcompletePathConsole, "w");	// to save analysis data of each file
			
			if(fileCounter==1){
				DisplayCommonSettingsObsSeq(fp_obsseq_console);
				if(!codebook_universe_generation)
				{
					//to save observation sequence 
					sprintf(OSeqcompletePathFinOp, "%s%s %s/%s%s.txt", input_folder, WordFolderName, WordNames[d], lambda_obs_seq_file_name, WordNames[d]);  // {input_lamda/}+{WordFolderName}+" "+{1}+"/"+{obs_seq__}+{1}+".txt 
									
					fp_obsseq_final_op = fopen(OSeqcompletePathFinOp, "w"); //to save compelete observation sequence in one file
				}
			}
			if(fp_obs_seq_ip == NULL || fp_obs_seq_norm == NULL || fp_obsseq_norm_seg == NULL ||  fp_obsseq_console==NULL ){ 
						perror("\n Error: ");
						printf("\n File Names are : \n  %s, \n  %s, \n  %s, \n %s \n", OSeqcompletePathIp, OSeqcompletePathNorm, OSeqcompletePathNormSeg, OSeqcompletePathConsole  );
						getch();
						return 1;
			}
			
			
			
		if(fileCounter==1){
			printf("  ---->  FILE: %s,\n", OSeqcompletePathIp);  
		}
		else
		{
			printf("\t %s%d.txt,", voice_data_prefix, fileCounter);  
		}


		fprintf(fp_obsseq_console, "\n ----------------------- START - ANALYZING OF FILE: %s ----------------------- \n", OSeqcompletePathIp);

		// DC Shift and Normalizing
			normalize_dcshift_samples();

		// Frames ZCR and Energy. STE Marker
			zcr_energy_frames();

		   //if(segregate_speech){						//only if you want to segregate speech into separate file.
			// calculating noise energy and threshold.
				noiseEnergy_thresholds_frames();						// if you want to calculate thresholds for zcr and energy
					
				marker_start_end_segregated();							//this and above func, if you want to detect start, end marker of speech, and to save it in separate file.
				fclose(fp_obsseq_norm_seg);	// closing file stream
			//}
		 //  else
		 //  {
			//   fclose(fp_obsseq_norm_seg);		// closing file stream
			//   remove(OSeqcompletePathNormSeg);		//removing unnecessory file created.
		 //  }
			if(!segregate_speech)
			{
				remove(OSeqcompletePathNormSeg);		//removing unnecessory
			}

		  // closing file stream, as no longer needed.
		   fflush(fp_obs_seq_ip); fclose(fp_obs_seq_ip); 
		   fflush(fp_obs_seq_norm); fclose(fp_obs_seq_norm);
		   remove(OSeqcompletePathNorm);	//comment it if you want to keep normalised data file.

		// Calculating Coefficients for Voiced Frames of File
			long totFramesVoice = end-start+1;
			calculateCoefficientsForFramesOfSpeech(totFramesVoice); //for each frame calculate coefficients
			
			if(codebook_universe_generation)
			{
				for(int ff=0; ff<totFramesVoice; ff++){
					for(int i=1;i<=p;i++){
						fprintf(fp_obsseq_final_op, "%lf,", C_rsw[ff][i]);
					}
					fprintf(fp_obsseq_final_op, "\n");
				}
			}
			else
			{
				generateObeservationSequence(totFramesVoice);	//for each frame calculate codebook index
				print_observation_sequence(totFramesVoice,  d);		//print observation seq in file
			}

				//printf("\n ----------------------- END Analyzing OF File: %s ----------------------- \n", OSeqfileNameIp);  
				fprintf(fp_obsseq_console, "\n ----------------------- END - ANALYZING OF FILE: %s ----------------------- \n", OSeqfileNameIp);
			
			// closing  stream, as no longer needed.
			fflush(fp_obsseq_console); fclose(fp_obsseq_console);
		}//end of filecounter loop -------------------------------------------------------------------------------------------------------------------
	
	if(!codebook_universe_generation)
	{
		printf("\n\n  ----> Digit %s:: \n\t Observation Sequence File Generated: %s\n\n", WordNames[d], OSeqcompletePathFinOp); 
		printf("\n-----------------------------------------------------\n");
			
		// closing file stream, as no longer needed.
		fflush(fp_obsseq_final_op); fclose(fp_obsseq_final_op);
	}
		//system("pause");
	}//end of digit loop ------------------------------------------------------------------------------------------------------------------------------
	
	if(codebook_universe_generation)
	{
		printf("\n\n\n  ----> CodeBook Universe File Generated: %s\n\n", OSeqcompletePathFinOp); 
		printf("\n-----------------------------------------------------\n");
		fflush(fp_obsseq_final_op); fclose(fp_obsseq_final_op);
	}
	
	return 0;
}//sequence_generation

/**************************************************************************************************
	Generate Observation Sequence For one Live Recording
**************************************************************************************************/
int live_sequence_generation()
{	
	printf("\n-----------------------------------------------------\n");
	printf("\n\t ---~---~---~---~---~--- GENERATING OBSERVATION SEQUENCE FOR (%s) ---~---~---~---~---~---\n", "LIVE VOICE RECORDING");
	/**************** Creating necessary file Path for data. ****************/
			
	//input file name
	sprintf(OSeqcompletePathIp, "%s%s.txt", input_live_voice_folder, liveRecordingFileName); 
	//segregated file data name 
	sprintf(OSeqcompletePathNorm, "%s%s_normalized_samples.txt", fileOutputRecordingNorSeg, liveRecordingFileName); 
	sprintf(OSeqcompletePathNormSeg, "%s%s_normalized_segregated_data.txt", fileOutputRecordingNorSeg, liveRecordingFileName); 
	//to save analysis file
	sprintf(OSeqcompletePathConsole, "%s%s_analysis.txt", fileOutputRecordingAnalysis, liveRecordingFileName);
	
	/**************** Opening respective files. ****************/
	fp_obs_seq_ip = fopen(OSeqcompletePathIp, "r");				//to read input file
	fp_obs_seq_norm = fopen(OSeqcompletePathNorm, "w+");		//to save normalised samples
	fp_obsseq_norm_seg = fopen(OSeqcompletePathNormSeg, "w");  //to save segregated recording from start to end
	fp_obsseq_console = fopen(OSeqcompletePathConsole, "w");	// to save analysis data of each file

	DisplayCommonSettingsObsSeq(fp_obsseq_console);
	//to save observation sequence 
	sprintf(OSeqcompletePathFinOp, "%s%s_obs_seq_.txt", input_live_voice_folder, liveRecordingFileName);  //		
	fp_obsseq_final_op = fopen(OSeqcompletePathFinOp, "w"); //to save compelete observation sequence in one file

	if(fp_obs_seq_ip == NULL || fp_obs_seq_norm == NULL ||  fp_obsseq_final_op==NULL || fp_obsseq_norm_seg == NULL ||  fp_obsseq_console==NULL ){ 
			perror("\n Error: ");
			printf("\n File Names are : \n  %s, \n  %s, \n  %s, \n  %s \n %s \n", OSeqcompletePathIp, OSeqcompletePathNorm, OSeqcompletePathNormSeg,  OSeqcompletePathFinOp, OSeqcompletePathConsole  );
			getch();
			return EXIT_FAILURE;
	}
			
	printf("\n  ----> ANALYZING OF FILE: %s", OSeqcompletePathIp);   
	
	fprintf(fp_obsseq_console, "\n ----------------------- START - ANALYZING OF FILE: %s ----------------------- \n", OSeqcompletePathIp);

	/**************** DC Shift and Normalizing ****************/
		normalize_dcshift_samples();

	/**************** Frames ZCR and Energy. STE Marker ****************/
		zcr_energy_frames();

		//if(segregate_Live_speech){						//only if you want to segregate speech into separate file.
		/****************  calculating noise energy and threshold. ****************/
			noiseEnergy_thresholds_frames();						// if you want to calculate thresholds for zcr and energy
					
		/**************** start and end marker of speech ****************/
			marker_start_end_segregated();							//this and above func, if you want to detect start, end marker of speech, and to save it in separate file.
			fclose(fp_obsseq_norm_seg);	// closing file stream
		//}
		//else
		//{
		//	fclose(fp_obsseq_norm_seg);		// closing file stream
		//	remove(OSeqcompletePathNormSeg);		//removing unnecessory file created.
		//}
		if(!segregate_Live_speech)
		{
			remove(OSeqcompletePathNormSeg);		//removing unnecessory
		}

	 // closing file stream, as no longer needed.
		fflush(fp_obs_seq_ip); fclose(fp_obs_seq_ip); 
		fflush(fp_obs_seq_norm); fclose(fp_obs_seq_norm);
		remove(OSeqcompletePathNorm);	//comment it if you want to keep normalised data file.

		/****************  Calculating Coefficients for Voiced Frames of File ****************/
			long totFramesVoice = end-start+1;
			calculateCoefficientsForFramesOfSpeech(totFramesVoice); //for each frame calculate coefficients
			generateObeservationSequence(totFramesVoice);	//for each frame calculate codebook index
			print_observation_sequence(totFramesVoice,  -1);		//print observation seq in file
				
				//printf("\n ----------------------- END Analyzing OF File: %s ----------------------- \n", OSeqfileNameIp);  
				fprintf(fp_obsseq_console, "\n ----------------------- END - ANALYZING OF FILE: %s ----------------------- \n", OSeqfileNameIp);
			
			// closing  stream, as no longer needed.
			fflush(fp_obsseq_console); fclose(fp_obsseq_console);

	
	
		printf("\n\n  ----> Live Observation Sequence File Generated: %s\n\n", OSeqcompletePathFinOp); 
		printf("\n-----------------------------------------------------\n");
			
		// closing file stream, as no longer needed.
		fflush(fp_obsseq_final_op); fclose(fp_obsseq_final_op);
		//system("pause");

	return 0;
}//LiveSequence



// hmm functions	
//#include "hmm_functions.h"

/**************************************************************************************
	To Read Lambda(A,B,Pi) and Observation Sequence from File
	Input: WordNames Array Index
**************************************************************************************/
void readLambdaABPi(int d, unsigned int model_type){
	FILE *fp_ind;
	char completePathInd[200]; 
	char completePathInput[200];
	char model_type_name[20];
	char lambda_folder_name[50];

	if(model_type==1) // use old model to read from input folder
	{
		sprintf(completePathInput, "%s%s %s/", input_folder, WordFolderName, WordNames[d]);
		sprintf(lambda_folder_name, "%s %s/", WordFolderName, WordNames[d]);
		strcpy(model_type_name, "Input Folder Model");
	}
	else if(model_type==2) //use new converged model to read from output folder
	{
		sprintf(completePathInput, "%s%s/%s/", output_folder, output_folder_Model_name, WordNames[d]); 
		sprintf(lambda_folder_name, "%s/%s/", output_folder_Model_name, WordNames[d]);
		strcpy(model_type_name, "Output Folder Model");
	}
	
	//printf("\n ----------------------- ----------------------- > Reading Lambda from File: %s %s (%s) < ----------------------- \n", WordFolderName, WordNames[d], model_type_name); 
	//fprintf(fp_console, "\n ----------------------- ----------------------- > Reading Lambda from File: %s %s (%s) < ----------------------- \n", WordFolderName, WordNames[d], model_type_name);				
		
	//fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Reading Lambda(A): %s/%s%s.txt\n\n", lambda_folder_name,LambdaFileNames[0], WordNames[d]);
		
		sprintf(completePathInd, "%s%s%s.txt", completePathInput,LambdaFileNames[0], WordNames[d]);  // {completePathInput}+{A_}+{1}+".txt 
		fp_ind = fopen(completePathInd, "r");				//to read input A from lamda
			if(fp_ind == NULL){ 
				perror("\nError: ");
				printf("\n File Name is: %s\n", completePathInd);
				getch();
				return;
			}	
		for (int si = 0; si < N; ++si){
			//fprintf(fp_console, "S[%d]\t", si+1);
			for (int sj = 0; sj < N; ++sj){
					fscanf(fp_ind,"%Lf",&A[si][sj]);
					//fprintf(fp_console, "%.16g(%d)\t",A[si][sj], sj+1);
				}
			//fprintf(fp_console, "\n");
		}
		//for (int i = 0; i < N; ++i)
		//{
		//	for (int j = 0; j < N; ++j)
		//		printf("%0.20g\t", A[i][j]);
		//	printf("\n");
		//}
		fflush(fp_ind);  fclose(fp_ind);


	//fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Reading Lambda(B): %s/%s%s.txt\n", lambda_folder_name,LambdaFileNames[1], WordNames[d]);
		sprintf(completePathInd, "%s%s%s.txt", completePathInput,LambdaFileNames[1], WordNames[d]);  // {completePathInput}+{B_}+{1}+".txt 
		fp_ind = fopen(completePathInd, "r");				//to read input B from lamda
			if(fp_ind == NULL){ 
				perror("\nError: ");
				printf("\n File Name is: %s\n", completePathInd);
				getch();
				return;
			}	
		
		for (int si = 0; si < N; ++si){
			//fprintf(fp_console, "S[%d]\t", si+1);
			for (int m = 0; m < M; ++m){
					fscanf(fp_ind,"%Lf",&B[si][m]);
					//fprintf(fp_console, "%.16g(%d)\t",B[si][m], m+1);
				}
			//fprintf(fp_console, "\n");
		}
		//for (int i = 0; i < N; ++i)
		//{
		//	for (int j = 0; j < M; ++j)
		//		printf("%.20g\t", B[i][j]);
		//	printf("\n");
		//}	
		fflush(fp_ind);  fclose(fp_ind); 
				
	//fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Reading Lambda(PI): %s/%s%s.txt\n",lambda_folder_name, LambdaFileNames[2], WordNames[d]);
		sprintf(completePathInd, "%s%s%s.txt", completePathInput,LambdaFileNames[2], WordNames[d]);  // {completePathInput}+{Pi_}+{1}+".txt 
		fp_ind = fopen(completePathInd, "r");				//to read input Pi from lamda	
			if(fp_ind == NULL){ 
				perror("\nError: ");
				printf("\n File Name is: %s\n", completePathInd);
				getch();
				return;
			}	
			
		for (int si = 0; si < N; ++si){
			fscanf(fp_ind,"%Lf",&PI[si]);
			//fprintf(fp_console, "%0.16g(S%d)\t",PI[si], si+1);
		}
		//fprintf(fp_console, "\n");

		fflush(fp_ind); fclose(fp_ind);
			
		//printf("\n -----------------------> Reading Completed: %s %s <----------------------- \n", WordFolderName, WordNames[d]); 
}//readLambdaABPi

void readObsSeq(int d, unsigned short int seq_type){
	
	unsigned short NumOfSequences;
	char lambda_obs_seq_file_name[50];
	char seq_type_name[20];

	FILE *fp_obs ;			//to read observation seq
	char completePathObs[200] ;

	if(seq_type == 1)
	{	
		NumOfSequences = On;
		strcpy(lambda_obs_seq_file_name, LambdaFileNames[3]);
		strcpy(seq_type_name, "TRAINING");
	}
	else if(seq_type == 2)
	{	
		NumOfSequences = Ot;
		strcpy(lambda_obs_seq_file_name, LambdaFileNames[4]);
		strcpy(seq_type_name, "TESTING");
	}

	
	sprintf(completePathObs, "%s%s %s/%s%s.txt", input_folder, WordFolderName, WordNames[d], lambda_obs_seq_file_name, WordNames[d]);  // {input_lamda/}+{WordFolderName}+" "+{1}+"/"+{obs_seq__}+{1}+".txt 

	fp_obs = fopen(completePathObs, "r"); //to save compelete observation sequence in one file
	//printf("\n File Names is :  %s \n", completePathObs);
	if(fp_obs == NULL ){ 
		perror("\n Error: ");
		printf("\n No Observation Sequence: File Names is :  %s \n", completePathObs);
		printf("\n\t Therfore Generating Sequence Now: \n");
		sequence_generation(seq_type);
		getch();
		//return;
	}
		
	printf("\n ---#---#---#---#---#---#---#---> Reading %s Observation Sequence From File: %s %s/%s%s.txt <---#---#---#---#---#---#---#--- \n", seq_type_name, WordFolderName, WordNames[d], lambda_obs_seq_file_name, WordNames[d]); 
			fprintf(fp_console, "\n ---#---#---#---#---#---#---#---> Reading %s Observation Sequence From File: %s %s/%s%s.txt <---#---#---#---#---#---#---#--- \n", seq_type_name, WordFolderName, WordNames[d], lambda_obs_seq_file_name, WordNames[d]);

		fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> %s Obs Seq: %s %s/%s%s.txt \n", seq_type_name, WordFolderName, WordNames[d],lambda_obs_seq_file_name, WordNames[d]);
				char skipDashes[1024];
				 int dd=0;
				  int num_frames_in_seq=0;
				for (int i = 0; i < NumOfSequences; ++i){

					fscanf(fp_obs,"%s %d:%d %s",&skipDashes,&dd, &num_frames_in_seq, &skipDashes);
					OFmax[i] = num_frames_in_seq;	// store max frames of each sequence

					fprintf(fp_console, "O[%d]:%i:\t", i+1,OFmax[i]);

					for (int t = 0; t < num_frames_in_seq; ++t){
							fscanf(fp_obs,"%d",&O[i][t]);
							fprintf(fp_console, "%d (%d)\t",O[i][t], t+1);
						}
					fprintf(fp_console, "\n");

					
				}// for each sequences
				
		printf("\n -----------------------> Reading Completed %s Observation Sequence: %s %s/%s%s.txt <----------------------- \n", seq_type_name, WordFolderName, WordNames[d], lambda_obs_seq_file_name, WordNames[d]); 
		printf("\n-----------------------------------------------------\n");
}//readTrainingTestingObsSeq

void readObsSeqOfLiveRecording(){

	unsigned short NumOfSequences=1;

	FILE *fp_obs ;			//to read observation seq
	char completePathObs[200] ;

	sprintf(completePathObs, "%s%s_obs_seq_.txt", input_live_voice_folder, liveRecordingFileName); 

	fp_obs = fopen(completePathObs, "r"); //to save compelete observation sequence in one file
	//printf("\n File Names is :  %s \n", completePathObs);
	if(fp_obs == NULL ){ 
		perror("\n Error: ");
		printf("\n No Observation Sequence: File Names is :  %s \n", completePathObs);
		printf("\n\t Therfore Generating Sequence Now: \n");
		live_sequence_generation();
		getch();
		//return;
	}
		
	printf("\n ---~---> Reading %s Observation Sequence From File: %s <---~--- \n", "LIVE VOICE RECORDING", completePathObs); 
		fprintf(fp_console, "\n ---~---> Reading %s Observation Sequence From File: %s <---~--- \n", "LIVE VOICE RECORDING", completePathObs);
		char skipDashes[1024];
			int dd=0;
			int num_frames_in_seq=0;
		for (int i = 0; i < NumOfSequences; ++i){

			fscanf(fp_obs,"%s %d:%d %s",&skipDashes,&dd, &num_frames_in_seq, &skipDashes);
			OFmax[i] = num_frames_in_seq;	// store max frames of each sequence

			fprintf(fp_console, "O[%d]:%i:\t", i+1,OFmax[i]);

			for (int t = 0; t < num_frames_in_seq; ++t){
					fscanf(fp_obs,"%d",&O[i][t]);
					fprintf(fp_console, "%d (%d)\t",O[i][t], t+1);
				}
			fprintf(fp_console, "\n");
		}// for each sequences
				
		//printf("\n ---~---~---~---~---~---> Reading Completed %s Observation Sequence From File: %s <---~---~---~---~---~--- \n", "LIVE VOICE RECORDING", completePathObs); 
		printf("\n-----------------------------------------------------\n");
}//readTrainingTestingObsSeq
/**************************************************************************************
	P1 Evaluation Problem (Scoring Problem) | SOLUTION: FORWARD PROCEDURE
	Input: Observation Sequnce, Observation Count
	Output: Probability(Observation sequence given lambda)
**************************************************************************************/
long double P1_Forward_Procedure(int *Oi, int o){
	long double alpha_t_sum =0;
	long double prob_alpha=0;

	int T = OFmax[o];
	//S1: Intialization //t=0 (or t1)
	for(int s=0; s<N; ++s)
		alpha[0][s] = PI[s]*B[s][Oi[0]-1];

	//S2: Induction
	for(int t=0; t<T-1; ++t){
		for(int sj=0; sj<N; ++sj){
				alpha_t_sum=0;
			for(int si=0; si<N; ++si){
					alpha_t_sum+=alpha[t][si]*A[si][sj];
			}// for si
			 alpha[t+1][sj] = alpha_t_sum*B[sj][Oi[t+1]-1];
		}//for sj
	}//for t

	//S3: Termination
	for(int s=0; s<N; ++s)
		 prob_alpha += alpha[T-1][s];

	//printing in file
	//fprintf(fp_console, "\n -----------------------> ALPHA MATRIX for ~~~~~~~~~~~~~ O[%d]\n", o+1);
	//for (int s = 0; s < N; ++s)
	//{
		//fprintf(fp_console, "S[%d]:: \t", s+1);
		//for (int t = 0; t < T; ++t)
		//{
			//fprintf(fp_console, "%.5g(%d)\t",alpha[t][s],t+1);
		//}
		//fprintf(fp_console, "\n");
	//}
	//fprintf(fp_console, "\n :::: Alpha Probabiltiy :  %g\n", prob_alpha);

	probability_alpha = prob_alpha;
	return prob_alpha;
}//P1_Forward_Procedure

void PRINT_P1_Forward_Procedure(int o){
	fprintf(fp_console, "\n -----------------------> ALPHA MATRIX for ~~~~~~~~~~~~~ O[%d]\n", o+1);
	int T = OFmax[o];
	for (int s = 0; s < N; ++s)
	{
		fprintf(fp_console, "S[%d]:: \t", s+1);
		for (int t = 0; t < T; ++t)
		{
			fprintf(fp_console, "%.5g(%d)\t",alpha[t][s],t+1);
		}
		fprintf(fp_console, "\n");
	}
	fprintf(fp_console, "\n :::: Alpha Probabiltiy :  %g\n", probability_alpha);

}//PRINT_P1_Forward_Procedure

/**************************************************************************************
	P1 Evaluation Problem (Scoring Problem) | SOLUTION: BACKWARD PROCEDURE
	Input: Observation Sequnce, Observation Count
	Output: Probability(Observation sequence given lambda)
**************************************************************************************/
long double P1_Backward_Procedure(int *Oi, int o){
	long double beta_t_sum =0;
	long double prob_beta=0;
	int T = OFmax[o];
	//S1: Intialization //t=0 (or t1)
	for(int s=0; s<N; ++s)
		beta[T-1][s] = 1;
	
	//S2: Induction
	for(int t=T-2; t>=0; --t){
		for(int si=0; si<N; ++si){
				beta_t_sum=0;
			for(int sj=0; sj<N; ++sj){
					beta_t_sum += A[si][sj]*B[sj][Oi[t+1]-1]*beta[t+1][sj];
			}// for sj
			 beta[t][si] = beta_t_sum;
		}//for si
	}//for t

	//S3: Termination
	for(int s=0; s<N; ++s)
		 prob_beta += beta[0][s];

	//printing in file
	//fprintf(fp_console, "\n -----------------------> BETA MATRIX for ~~~~~~~~~~~~~ O[%d]\n", o+1);
	//for (int s = 0; s < N; ++s)
	//{
	//	fprintf(fp_console, "S[%d]:: \t", s+1);
	//	for (int t = 0; t < T; ++t)
	//	{
	//		fprintf(fp_console, "%.5g(%d)\t",beta[t][s],t+1);
	//	}
	//	fprintf(fp_console, "\n");
	//}

	//fprintf(fp_console, "\n :::: Beta Probabiltiy :  %g\n", prob_beta);
	probability_beta=prob_beta;
	return prob_beta;
}//P1_Backward_Procedure

void PRINT_P1_Backward_Procedure(int o){
	fprintf(fp_console, "\n -----------------------> BETA MATRIX for ~~~~~~~~~~~~~ O[%d]\n", o+1);
	int T = OFmax[o];
	for (int s = 0; s < N; ++s)
	{
		fprintf(fp_console, "S[%d]:: \t", s+1);
		for (int t = 0; t < T; ++t)
		{
			fprintf(fp_console, "%.5g(%d)\t",beta[t][s],t+1);
		}
		fprintf(fp_console, "\n");
	}

	fprintf(fp_console, "\n :::: Beta Probabiltiy :  %g\n", probability_beta);

}//PRINT_P1_Backward_Procedure

/**************************************************************************************
	P2 Uncovering the Problem | SOLUTION: VITERBI ALGO
	Input: Observation Sequnce, Observation Count
**************************************************************************************/
void P2_Viterbi_Algo(int *Oi, int o){
	
	//long double temp_delta=0, temp_delta_max=0;
	int argmax=0; //argument which gives the max probability of delta
	int T = OFmax[o];
	//S1: Intialization //t=0 (or t1)
	for(int s=0; s<N; ++s){
		delta[0][s] = PI[s]*B[s][Oi[0]-1];
		psi[0][s]=-1;
	}

	//S2: Recursion
	for(int t=1; t<T; ++t){
		
		for(int sj=0; sj<N; ++sj){
				argmax=0;				// argument i for which delta[t][j] is maximum, let first state is max
			for(int si=1; si<N; ++si){
				//temp_delta = delta[t-1][si]*A[si][sj];
				//temp_delta_max = delta[t-1][argmax]*A[argmax][sj];
					if((delta[t-1][si]*A[si][sj]) > (delta[t-1][argmax]*A[argmax][sj]))
						argmax=si;
			}// for si
			 delta[t][sj] = delta[t-1][argmax]*A[argmax][sj]*B[sj][Oi[t]-1];
			 psi[t][sj]=argmax;		//argument index which gave the max probability.
		}//for sj
	
	}//for t
	
	//S3: Termination
	argmax=0;
	for(int sj=1; sj<N; ++sj){
			if(delta[T-1][sj] > delta[T-1][argmax])
				argmax=sj;
	}//for sj
	Pstar = delta[T-1][argmax];
	Qstar[T-1]=argmax;		//argument index which gave the max probability.

	//S4: Backtracking, State Sequence Path.
	for(int t=T-2; t>=0; --t)
	{
		Qstar[t]=psi[t+1][Qstar[t+1]];  
	}//for t

	//fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Value of Pstar: %g\n", Pstar);
	//fprintf(fp_console, ":::::O[%d]:\t",o+1);
	//for (int t = 0; t < T; ++t){
	//		fprintf(fp_console, "%d (%d)\t",Oi[t], t+1);
	//}	
	//fprintf(fp_console, "\n:::::Q[%d]:\t",o+1);
	//for (int t = 0; t < T; ++t){
	//		fprintf(fp_console, "%d (%d)\t",Qstar[t]+1, t+1);
	//}
	//fprintf(fp_console, "\n");	

}//P2_Viterbi_Algo

void PRINT_P2_Viterbi_Algo(int o){

	int T = OFmax[o];

	//printing in file
	fprintf(fp_console, "\n -----------------------> DELTA & PSI MATRIX ~~~~~~~~~~~~~ O[%d]\n", o+1);
	for (int s = 0; s < N; ++s)
	{
		fprintf(fp_console, "S[%d]:: \t", s+1);
		for (int t = 0; t < T; ++t)
		{
			fprintf(fp_console, "%.5g(%d)\t",delta[t][s],t+1);
		}
		fprintf(fp_console, "\n");
	}
	fprintf(fp_console, "\n");
	for (int s = 0; s < N; ++s)
	{
		fprintf(fp_console, "Psi[%d]\t", s+1);
		for (int t = 0; t < T; ++t)
		{
			fprintf(fp_console, "%d(%d)\t",psi[t][s]+1,t+1);
		}
		fprintf(fp_console, "\n");
	}

}//PRINT_P2_Viterbi_Algo

void PRINT_P2_PStar_State_Sequence(int *Oi, int o){
	int T = OFmax[o];
	fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Value of Pstar: %g\n", Pstar);
	fprintf(fp_console, ":::::O[%d]:\t",o+1);
	for (int t = 0; t < T; ++t){
			fprintf(fp_console, "%d (%d)\t",Oi[t], t+1);
	}	
	fprintf(fp_console, "\n:::::Q[%d]:\t",o+1);
	for (int t = 0; t < T; ++t){
			fprintf(fp_console, "%d (%d)\t",Qstar[t]+1, t+1);
	}
	fprintf(fp_console, "\n");
}//PRINT_P2_PStar_State_Sequence

/**************************************************************************************************
	P2 Uncovering the Problem | SOLUTION: GAMMA PRCOEDURE 
	Probability of being in state Si at time t given the observation sequnece and model lambda
**************************************************************************************************/
long double P2_Gamma_Procedure(int o){

	int T = OFmax[o];

	int Q[Tmax];	 // Gamma Procedure, Best State Sequence Path which is most likely state for each time point t
	int argmax=0;
	long double denominator_sum=0;
	long double prob_gamma=1;

	for(int t=0; t<T; t++){

		denominator_sum=0;
		for(int si=0; si<N; si++)
			denominator_sum += alpha[t][si]*beta[t][si]; 
		
		argmax=0;
		for(int si=0; si<N; si++){
			gamma[t][si] = alpha[t][si]*beta[t][si]/denominator_sum;
			if(gamma[t][si]>gamma[t][argmax])
				argmax=si;
		}//for si
		Q[t]=argmax;
	}// for t

	for(int t=0; t<T; t++)
		prob_gamma*=gamma[t][Q[t]];

	return prob_gamma;
}//P2_Gamma_Procedure

void PRINT_P2_Gamma_Procedure(int o){
	int T = OFmax[o];
	//printing in file
	fprintf(fp_console, "\n -----------------------> GAMMA MATRIX ~~~~~~~~~~~~~ O[%d]\n", o+1);
	for (int s = 0; s < N; ++s)
	{
		fprintf(fp_console, "S[%d]:: \t", s+1);
		for (int t = 0; t < T; ++t)
		{
			fprintf(fp_console, "%.5g(%d)\t",gamma[t][s],t+1);
		}
		fprintf(fp_console, "\n");
	}
	fprintf(fp_console, "\n");

}//PRINT_P2_Gamma_Procedure

/**************************************************************************************************
	P3 Re-estimation Problem | SOLUTION: Baum Welch Method
	XI: Prob of being in state Si at time t and state Sj at time t+1 given the model lambda
		and observation sequence.
	Input: Observation Sequnce, Observation Count
**************************************************************************************************/
void P3_Baum_Welch_Procedure(int *Oi, int o){
	int T = OFmax[o];
	long double denominator_sum=0; 

	for(int t=0; t<T-1; t++) // For all T-1 
	{
		denominator_sum=0;
		for(int si=0; si<N; si++){
			for(int sj=0; sj<N; sj++)
				denominator_sum += alpha[t][si]*A[si][sj]*B[sj][Oi[t+1]-1]*beta[t+1][sj];
		}
		for(int si=0; si<N; si++){
			for(int sj=0; sj<N; sj++)
				XI[t][si][sj]=(alpha[t][si]*A[si][sj]*B[sj][Oi[t+1]-1]*beta[t+1][sj])/denominator_sum;
		}
	}
}//P3_Baum_Welch_Procedure

/**************************************************************************************************
	P3 Re-estimation Procedure | Calculating Abar Bbar Pibar 
	XI: Prob of being in state Si at time t and state Sj at time t+1 given the model lambda
		and observation sequence.
	Input: Observation Sequnce, Observation Count
**************************************************************************************************/
void P3_Reestimation_Procedure(int *Oi, int o){

	int T = OFmax[o];

	// Re-estimation of Pi as Pi_bar
	for(int si=0; si<N; si++) {
		PI_BAR[si]=gamma[0][si];
	}//Pi as Pi_bar

	// Re-estimation of A as A_bar
	long double nume_exp_num_of_transitionf_from_si_to_sj=0;
	long double deno_exp_num_of_transitionf_from_si=0;

	for(int si=0; si<N; si++) 
	{
		for(int sj=0; sj<N; sj++)
		{
			nume_exp_num_of_transitionf_from_si_to_sj=0;
			deno_exp_num_of_transitionf_from_si=0;
			for(int t=0; t<T-2; t++)  // from (1 to T-1)
			{
				nume_exp_num_of_transitionf_from_si_to_sj += XI[t][si][sj];
				deno_exp_num_of_transitionf_from_si += gamma[t][si];
			}

			A_BAR[si][sj]=nume_exp_num_of_transitionf_from_si_to_sj/deno_exp_num_of_transitionf_from_si;
		}
	}//A as A_bar

	// Re-estimation of B as B_bar
	long double nume_exp_num_of_times_in_sj_observing_vk =0;
	long double deno_exp_num_of_times_in_sj =0;

	for(int sj=0; sj<N; sj++)
	{
		for(int k=0; k < M; k++)  // obs seq value are from 1 to M
		{
			nume_exp_num_of_times_in_sj_observing_vk=0;
			deno_exp_num_of_times_in_sj=0;
			for(int t=0; t<T; t++)
			{
				if(Oi[t]-1==k)
					nume_exp_num_of_times_in_sj_observing_vk += gamma[t][sj];

				deno_exp_num_of_times_in_sj+=gamma[t][sj];
			}
			B_BAR[sj][k]=nume_exp_num_of_times_in_sj_observing_vk/deno_exp_num_of_times_in_sj;
		}
	}// B as B_bar


}//P3_Reestimation_Procedure

/**************************************************************************************************
	Print Model Lamda BAR in the File. (or Display on console)
**************************************************************************************************/
void print_model_lambda_bar(){

	fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Lambda(A):\n");
		for (int si = 0; si < N; ++si){
			fprintf(fp_console, "S[%d]\t", si+1);
			for (int sj = 0; sj < N; ++sj){
					//printf("%0.16g\t", A[si][sj]);
					fprintf(fp_console, "%.16g (%d)\t",A_BAR[si][sj], sj+1);
				}
			fprintf(fp_console, "\n");
		}

	fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Lambda(B):\n");
		for (int si = 0; si < N; ++si){
			fprintf(fp_console, "S[%d]\t", si+1);
			for (int m = 0; m < M; ++m){
					//printf("%.16g\t", B[si][m]);
					fprintf(fp_console, "%.16g (%d)\t",B_BAR[si][m], m+1);
				}
			fprintf(fp_console, "\n");
		}
		
				
	fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Lambda(PI):\n");
			for (int si = 0; si < N; ++si){
				fprintf(fp_console, "%.16g (S%d)\t",PI_BAR[si], si+1);
			}
			fprintf(fp_console, "\n");

}

/**************************************************************************************************
	Making Lambda Stochastic by adjusting values in max value in row.
**************************************************************************************************/
void make_lambda_stochastic(){
	long double row_sum = 0;
	long double row_max =0;
	int max_index = 0;

	// making A Stochastic
	for(int si=0; si<N; si++){
		row_sum=0;
		row_max = 0;
		max_index =-1;		

		for(int sj=0; sj<N; sj++){
			if( A_BAR[si][sj] > row_max )
			{	max_index = sj;
				row_max = A_BAR[si][sj];
			}

			row_sum += A_BAR[si][sj];
		}//row sum
		//if(row_sum != 1){
		//	A_BAR[si][max_index] = (row_sum > 1) ? (A_BAR[si][max_index] - (row_sum-1) ):(A_BAR[si][max_index] + (1-row_sum));
		//}
		A_BAR[si][max_index] -= (row_sum-1); 
		
	}// making A Stochastic

	// making B Stochastic
	for(int sj=0; sj<N; sj++){
		row_sum=0;
		row_max = 0;
		max_index =0;	

		for(int k=0; k<M; k++){
		
			if(B_BAR[sj][k] == 0)
				B_BAR[sj][k] = 1e-030;

			if( B_BAR[sj][k] > row_max )
			{	max_index = k;	
				row_max =  B_BAR[sj][k];
			}

			row_sum += B_BAR[sj][k];

		}
		//if(row_sum != 1){
		//	B_BAR[sj][max_index] = (row_sum > 1) ? (B_BAR[sj][max_index] - (row_sum-1) ):(B_BAR[sj][max_index] + (1-row_sum));
		//}
		B_BAR[sj][max_index] -= (row_sum-1);
	}// making B Stochastic

}//make_lambda_stochastic

/**************************************************************************************************
	Replace Old Model Lambda with New Model LambdaBar after Re-estimation
**************************************************************************************************/
void replace_old_model(){

	for(int si=0; si<N; si++) 
        PI[si]=PI_BAR[si];

    for(int si=0; si < N; si++) //assign the given values for transition probability distribution
        for(int sj=0; sj<N; sj++)
            A[si][sj]=A_BAR[si][sj];

    for(int sj=0; sj<N; sj++) //assign the given values for observation symbol probability distribution
        for(int k=0; k<M; k++)
            B[sj][k]=B_BAR[sj][k];


}//replace_old_model

/**************************************************************************************************
	Print Model Lamda in the File. (or Display on console)
**************************************************************************************************/
void print_model_lambda(){

	fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Lambda(A):\n");
		for (int si = 0; si < N; ++si){
			fprintf(fp_console, "S[%d]\t", si+1);
			for (int sj = 0; sj < N; ++sj){
					//printf("%0.16g\t", A[si][sj]);
					fprintf(fp_console, "%.16g (%d)\t",A[si][sj], sj+1);
				}
			fprintf(fp_console, "\n");
		}

	fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Lambda(B):\n");
		for (int si = 0; si < N; ++si){
			fprintf(fp_console, "S[%d]\t", si+1);
			for (int m = 0; m < M; ++m){
					//printf("%.16g\t", B[si][m]);
					fprintf(fp_console, "%.16g (%d)\t",B[si][m], m+1);
				}
			fprintf(fp_console, "\n");
		}
		
				
	fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Lambda(PI):\n");
			for (int si = 0; si < N; ++si){
				fprintf(fp_console, "%.16g (S%d)\t",PI[si], si+1);
			}
			fprintf(fp_console, "\n");

}

/**************************************************************************************************
	Feed Forward Model | Bakes Model
**************************************************************************************************/
void initialize_feed_forward_model()
{
	/**************** PI ****************/
	PI[0]=1.0; // first state
	for(int si=1; si<N; si++){
		PI[si]=0;	// rest of the state
	}

	/**************** A ****************/
	for(int si=0; si<N; si++)
	{
		for(int sj=0; sj<N; sj++)
		{
			if(si==sj)
				A[si][sj] = 0.8; // Prob of being in same state
			else if(si+1==sj)
				A[si][sj] = 0.2; // Prob to shift in next state
			else A[si][sj] = 0;
		}
	}
	A[N-1][N-1] = 1;		// Forcing to be in same state reaching final state

	/**************** B ****************/
	for(int sj=0; sj<N; sj++)
		for(int k=0; k<M; k++)
			B[sj][k] = 1.0/M;


}//initialize_feed_forward_model

/**************************************************************************************************
	Using Previous Converged Model as starting point
**************************************************************************************************/
void initialize_converged_model()
{
	/**************** PI ****************/
	PI[0]=1.0; // first state
	for(int si=1; si<N; si++){
		PI[si]=0;	// rest of the state
	}

	/**************** A ****************/
	for(int si=0; si<N; si++)
	{
		for(int sj=0; sj<N; sj++)
		{
			A[si][sj] = A_Prev[si][sj];
		}
	}

	/**************** B ****************/
	for(int sj=0; sj<N; sj++)
		for(int k=0; k<M; k++)
			B[sj][k] = B_Prev[sj][k];


}//initialize_feed_forward_model

/**************************************************************************************************
	Saving Converged Model to their Individual files
	Input: digit value
**************************************************************************************************/
void output_lambdaABPi_to_each_file(int d){
	FILE *fp_ind;
	char completePathInd[200];

// Save PI
	sprintf(completePathInd, "%s%s/%s/%s%s.txt", output_folder, output_folder_Model_name, WordNames[d], LambdaFileNames[2], WordNames[d]);  
	fp_ind = fopen(completePathInd, "w");				//to read input codebook
	if(fp_ind == NULL){ 
		perror("\n Error: ");
		printf("\n File Name is: %s", completePathInd);
		getch();
		return;
	}

	for (int si = 0; si < N; ++si){
		fprintf(fp_ind, "%.16g\t",PI[si]);
	}

	fflush(fp_ind); fclose(fp_ind); 

//Save A
	sprintf(completePathInd, "%s%s/%s/%s%s.txt", output_folder, output_folder_Model_name, WordNames[d], LambdaFileNames[0], WordNames[d]);  
	fp_ind = fopen(completePathInd, "w");				//to read input codebook
	if(fp_ind == NULL){ 
		perror("\n Error: ");
		printf("\n File Name is: %s", completePathInd);
		getch();
		return;
	}
	for (int si = 0; si < N; ++si){
		for (int sj = 0; sj < N; ++sj){
			fprintf(fp_ind, "%.16g\t",A[si][sj]);
		}
		fprintf(fp_ind, "\n");
	}
	fflush(fp_ind); fclose(fp_ind); 


//Save B
	sprintf(completePathInd, "%s%s/%s/%s%s.txt", output_folder, output_folder_Model_name, WordNames[d], LambdaFileNames[1], WordNames[d]);  
	fp_ind = fopen(completePathInd, "w");				//to read input codebook
	if(fp_ind == NULL){ 
		perror("\n Error: ");
		printf("\n File Name is: %s", completePathInd);
		getch();
		return;
	}
	for (int si = 0; si < N; ++si){
		for (int m = 0; m < M; ++m){
		
			fprintf(fp_ind, "%.16g\t",B[si][m]);
		}
		fprintf(fp_ind, "\n");
	}
	fflush(fp_ind); fclose(fp_ind); 

}

/**************************************************************************************************
	Convergence Procedure: use each utterance and then converging until Pstar improved
	then average model of all the converged models
**************************************************************************************************/
void covergence_procedure(){

	int skip=1;				// number of times ignore (Pstar_new > Pstar_old) and check once again for more Pstar_new

	for(int rc=1; rc<=repeatConvergence;rc++)//repeat using converged model second time
		{ 
			for(int o=0; o<On; ++o){
				
				if(rc==1){
					printf("\n  -###-###-###-###-###-###->>> Bakis Model | Observation O[%d] < ----------------------- ----------------------- \n", o+1);	
					fprintf(fp_console, "\n  -###-###-###-###-###-###->>> Bakis Model | Observation O[%d] < ----------------------- ----------------------- \n", o+1);	
					initialize_feed_forward_model();
					skip=1;
				}
				else {
					printf("\n -###-###-###-###-###-###->>> Converged Model | Observation O[%d] < ----------------------- ----------------------- \n", o+1);	
					fprintf(fp_console, "\n -###-###-###-###-###-###->>> Converged Model | Observation O[%d] < ----------------------- ----------------------- \n", o+1);	
					initialize_converged_model();
					skip=0;
				}

				printf("\n");
			
				int itr=1;
				do
				{
					P1_Forward_Procedure(O[o],o);
					P1_Backward_Procedure(O[o],o);
					P2_Gamma_Procedure(o);
					P2_Viterbi_Algo(O[o],o);

					if(showAlphaBetaPstarInConsole){
						printf("\n[%d]\t", itr); 
						printf("Alpha P = %g \t", probability_alpha);
						printf("Beta P = %g \t", probability_beta);
						printf("Pstar P = %g \t",Pstar);
					}
					P3_Baum_Welch_Procedure(O[o],o);
					P3_Reestimation_Procedure(O[o],o);
					make_lambda_stochastic();
					replace_old_model();

					itr++;
					Pstar_old = Pstar;
					P2_Viterbi_Algo(O[o],o);
					if(showAlphaBetaPstarInConsole){
						printf("New_Pstar P = %g\t",Pstar);
					}
				}while( (Pstar > Pstar_old && itr <= model_iterations) || (skip--));// while Pstar> Pstar_old loop

				//if(rc==3)
				//for(; itr<=model_iterations; itr++)
				//{
				//	P1_Forward_Procedure(O[o],o);
				//	P1_Backward_Procedure(O[o],o);
				//	P2_Gamma_Procedure(o);
				//	P2_Viterbi_Algo(O[o],o);

				//	if(showAlphaBetaPstarInConsole){
				//		printf("\n[%d]\t", itr); 
				//		printf("Alpha P = %g \t", probability_alpha);
				//		printf("Beta P = %g \t", probability_beta);
				//		printf("Pstar P = %g \t",Pstar);
				//	}
				//	P3_Baum_Welch_Procedure(O[o],o);
				//	P3_Reestimation_Procedure(O[o],o);
				//	make_lambda_stochastic();
				//	replace_old_model();

				//}//for each iterations itr

				// save converged lambda
				for (int si = 0; si < N; ++si){
					for (int sj = 0; sj < N; ++sj){
						converged_A[o][si][sj] = A[si][sj];
					}
				}

				for (int si = 0; si < N; ++si){
					for (int m = 0; m < M; ++m){
						converged_B[o][si][m] = B[si][m];
					}
				}

				printf(" Total Iterations: %d\t", itr-1); 
				printf(" Alpha P = %g \t", probability_alpha);
				printf( "Beta P = %g \t", probability_beta);
				printf( "Pstar P = %g \n",Pstar);
				
				fprintf(fp_console, "Total Iterations: %d\t", itr-1); 
				fprintf(fp_console, "Alpha P = %g \t", probability_alpha);
				fprintf(fp_console, "Beta P = %g \t", probability_beta);
				fprintf(fp_console, "Pstar P = %g \n",Pstar);
			}//for each observation seq 'o'

			// take averaged lambda
			long double lambda_sum=0;
			for (int si = 0; si < N; ++si){
				for (int sj = 0; sj < N; ++sj){
					lambda_sum=0;

					for(int u=0; u<On; u++)
					{
						lambda_sum += converged_A[u][si][sj];
					}

					A[si][sj] = lambda_sum/On ;
					A_Prev[si][sj] = A[si][sj] ;
				}
			}

			for (int si = 0; si < N; ++si){
				for (int m = 0; m < M; ++m){
					lambda_sum=0;
					for(int u=0; u<On; u++)
					{
						lambda_sum += converged_B[u][si][m];
					}
					B[si][m] = lambda_sum/On ;
					B_Prev[si][m] = B[si][m];
				}
			}
		
			fprintf(fp_console, "\n ----------------------- After Taking Average of Converged Lambdas of %d Training Sequence  \n", On);	
			
			if(showStateSeqAlphaBetaInFileForEachObsAfterConverge){
				for(int o=0; o<On; ++o){
					P2_Viterbi_Algo(O[o],o);
					PRINT_P2_PStar_State_Sequence(O[o],o);

						fprintf(fp_console, "\nO[%d]:\t",o+1);
						fprintf(fp_console, "Alpha P = %g\t",P1_Forward_Procedure(O[o],o));
						fprintf(fp_console, "Beta P = %g\t",P1_Backward_Procedure(O[o],o));
						fprintf(fp_console, "Pstar P = %g\t\n",Pstar);
				}
			}
			fprintf(fp_console, "\n ----------------------- -----------------------> Converged Lamda <----------------------- ----------------------- \n");	
			
			print_model_lambda();

		}// for each rc

}//covergence_procedure

/**************************************************************************************************
	Replace Model in Input Folder with Output Folder.
**************************************************************************************************/
void replace_old_models_files(){

	char from_lambda_location[300], to_lambda_location[300];
	FILE *fp_infile, *fp_outfile;
	long double temp;

	for(int d=0; d<W; d++){

		printf("\n ---#---#---#---#---#---#---#--->>> Replacing Model: %s %s <---#---#---#---#---#---#---#---\n", WordFolderName, WordNames[d]); 

// replace A
		sprintf(from_lambda_location, "%s%s/%s/%s%s.txt", output_folder, output_folder_Model_name, WordNames[d], LambdaFileNames[0], WordNames[d]);  // {output_folder/Models}+{0/}+{A_}+{1}+".txt 
		sprintf(to_lambda_location, "%s%s %s/%s%s.txt", input_folder, WordFolderName, WordNames[d], LambdaFileNames[0], WordNames[d]);  // {output_folder/Models}+{0/}+{A_}+{1}+".txt 
	
		fp_infile = fopen(from_lambda_location, "r");
		fp_outfile = fopen(to_lambda_location, "w");
			if(fp_infile == NULL || fp_outfile == NULL ){ 
				perror("\n Error: ");
				printf("\n File Names are: \nSrc: %s \nDest: %s", from_lambda_location, to_lambda_location);
				getch();
				return;
			}

		for (int si = 0; si < N; ++si){
			for (int sj = 0; sj < N; ++sj){
				fscanf(fp_infile, "%Lf",&temp);
				fprintf(fp_outfile, "%.16g\t",temp);
			}
			fprintf(fp_outfile, "\n");
		}

		fflush(fp_infile); fclose(fp_infile); 
		fflush(fp_outfile); fclose(fp_outfile); 

// replace B
		sprintf(from_lambda_location, "%s%s/%s/%s%s.txt", output_folder, output_folder_Model_name, WordNames[d], LambdaFileNames[1], WordNames[d]);  // {output_folder/Models}+{0/}+{B_}+{1}+".txt 
		sprintf(to_lambda_location, "%s%s %s/%s%s.txt", input_folder, WordFolderName, WordNames[d], LambdaFileNames[1], WordNames[d]);  // {output_folder/Models}+{0/}+{B_}+{1}+".txt 
		
		fp_infile = fopen(from_lambda_location, "r");
		fp_outfile = fopen(to_lambda_location, "w");
			if(fp_infile == NULL || fp_outfile == NULL ){ 
				perror("\n Error: ");
				printf("\n File Names are: \nSrc: %s \nDest: %s", from_lambda_location, to_lambda_location);
				getch();
				return;
			}

		for (int si = 0; si < N; ++si){
			for (int m = 0; m < M; ++m){
				fscanf(fp_infile, "%Lf",&temp);
				fprintf(fp_outfile, "%.16g\t",temp);
			}
			fprintf(fp_outfile, "\n");
		}
		fflush(fp_infile); fclose(fp_infile); 
		fflush(fp_outfile); fclose(fp_outfile); 

//replace pi
		sprintf(from_lambda_location, "%s%s/%s/%s%s.txt", output_folder, output_folder_Model_name, WordNames[d], LambdaFileNames[2], WordNames[d]);  // {output_folder/Models}+{0/}+{Pi_}+{1}+".txt 
		sprintf(to_lambda_location, "%s%s %s/%s%s.txt", input_folder, WordFolderName, WordNames[d], LambdaFileNames[2], WordNames[d]);  // {output_folder/Models}+{0/}+{Pi_}+{1}+".txt 
		fp_infile = fopen(from_lambda_location, "r");
		fp_outfile = fopen(to_lambda_location, "w");
			if(fp_infile == NULL || fp_outfile == NULL ){ 
				perror("\n Error: ");
				printf("\n File Names are: \nSrc: %s \nDest: %s", from_lambda_location, to_lambda_location);
				getch();
				return;
			}
		for (int si = 0; si < N; ++si){
			fscanf(fp_infile, "%Lf",&temp);
			fprintf(fp_outfile, "%.16g\t",temp);
		}

		fflush(fp_infile); fclose(fp_infile); 
		fflush(fp_outfile); fclose(fp_outfile);
	}// for each digit d<W

}

/**************************************************************************************************
	Test Offline Utterance of the Digits
**************************************************************************************************/
void offline_testing(int test_word, unsigned int model_type_to_use){

	long double cur_alpha_probability=0;
	long double max_probability=0;
	int word_index=-1;

	offline_correct_count=0;

	for(int u=0; u<Ot; u++){
					printf("\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Utterance: %s_%s/%s_O[%d] \n", WordFolderName, WordNames[test_word], LambdaFileNames[4], u+1);
						fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Utterance: %s_%s/%s_O[%d] \n", WordFolderName, WordNames[test_word], LambdaFileNames[4], u+1); 	 
			max_probability=0;
			cur_alpha_probability=0;
			word_index=-1;
		for(int w=0; w<W; w++){
			readLambdaABPi(w, model_type_to_use);
			cur_alpha_probability=P1_Forward_Procedure(O[u],u);

					printf("O[%d]:W[%s]::  Alpha P = %g\n", u+1, WordNames[w], cur_alpha_probability); 	 
					fprintf(fp_console, "O[%d]:W[%s]::  Alpha P = %g\n", u+1, WordNames[w], cur_alpha_probability); 	 

			if(cur_alpha_probability > max_probability){
				max_probability = cur_alpha_probability;
				word_index=w;
			}
		}//for each w word

		if(word_index==-1)
		{
			printf("\n -----------------> Actual digit: %s  ", WordNames[test_word]);
				printf("\n -----------------> Digit Recognized: %s\n", "NOT RECOGNIZED");
			fprintf(fp_console, "\n -----------------> Actual digit: %s  ", WordNames[test_word]);
			fprintf(fp_console, "\n -----------------> Digit Recognized: %s\n", "NOT RECOGNIZED");
		}
		else
		{
			printf("\n -----------------> Actual digit: %s  ", WordNames[test_word]);
				printf("\n -----------------> Digit Recognized: %s\n", WordNames[word_index]);
			fprintf(fp_console, "\n -----------------> Actual digit: %s  ", WordNames[test_word]);
			fprintf(fp_console, "\n -----------------> Digit Recognized: %s\n", WordNames[word_index]);
		}	
		
		if(strcmp(WordNames[test_word], WordNames[word_index])==0)offline_correct_count++;
	}// for each u utterance

	offline_overall_count +=offline_correct_count;

	fprintf(fp_console, "\n ------------------------------------------------------------------------\n"); 
}//offline_testing

/**************************************************************************************************
	Test Live Utterance of the Digits
**************************************************************************************************/
void live_testing(unsigned int model_type_to_use){

	
	/**************** Observation Sequence Generation ****************/
	live_sequence_generation();
	/**************** Read Observation Sequence ****************/
	readObsSeqOfLiveRecording();

	/**************** Testing ****************/
	int NumOfLiveUtterance=1;

	long double cur_alpha_probability=0;
	long double max_probability=0;
	int word_index=-1;

	offline_correct_count=0;

	for(int u=0; u<NumOfLiveUtterance; u++){
					printf("\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Live Utterance: %s_obs_seq_O[%d] \n", liveRecordingFileName, u+1);
						fprintf(fp_console, "\n ~~~~~~~~~~~~~~~~~~~~~~~~~> Live Utterance: %s_obs_seq_O[%d] \n", liveRecordingFileName, u+1); 	 
			max_probability=0;
			cur_alpha_probability=0;
			word_index=-1;
		for(int w=0; w<W; w++){
			readLambdaABPi(w, model_type_to_use);
			cur_alpha_probability=P1_Forward_Procedure(O[u],u);

					printf("O[%d]:W[%s]::  Alpha P = %g\n", u+1, WordNames[w], cur_alpha_probability); 	 
					fprintf(fp_console, "O[%d]:W[%s]::  Alpha P = %g\n", u+1, WordNames[w], cur_alpha_probability); 	 

			if(cur_alpha_probability > max_probability){
				max_probability = cur_alpha_probability;
				word_index=w;
			}
		}//for each w word
		if(word_index==-1)
		{
			printf("\n -----------------> Digit Recognized: %s\n", "NOT RECOGNIZED");
			fprintf(fp_console, "\n -----------------> Digit Recognized: %s\n", "NOT RECOGNIZED" );
		}
		else
		{
			printf("\n -----------------> Digit Recognized: %s\n", WordNames[word_index]);
			fprintf(fp_console, "\n -----------------> Digit Recognized: %s\n", WordNames[word_index]);
		}
		
	}// for each u utterance
	fprintf(fp_console, "\n ------------------------------------------------------------------------\n"); 
}//live_testing
/**************************************************************************************
	To Display Common Settings used in our System
	Input: File Pointer in case things needed to be written on file.
**************************************************************************************/
void DisplayCommonSettings(FILE *fp_set=NULL){
	// General Information to Display
	if(fp_set==NULL){
		printf("****-------- WELCOME TO HMM --------****\n");		
		printf("-Common Settings are : -\n");	
		printf(" P (=Q)(#of Cepstral Coefficients) : %d\n", p);
		printf(" Number of Words/Digits/HMM (W) : %d\n", W);	
		printf(" Number of States per HMM (N) : %d\n", N);	
		printf(" Number of Distinct Observation Symbols (M) or CodeBook Size (Y) : %d\n", M); 	
		printf(" Max Length of Observation Sequence (T) : %d\n", Tmax);			
		printf(" Number of Training Observations : %d\n", On);	
		printf(" Number of Testing Observations : %d\n", Ot);

		printf("\n");
		printf(" Frame Size : %d\n", sizeFrame);	
		printf(" Tokhura Weights : ");
		for(int i=0; i<q; i++){
			printf("%0.1f(%d) ", w_tkh[i],i+1);
		}
		printf("\n Amplitutde Value to Scale : %d\n", scaleAmp);			
		printf(" Intital Header Lines Ignore Count : %d\n", initIgnoreHeaderLines); 
		printf(" Intital Samples to Ignore : %d\n",initIgnoreSamples);	
		printf(" Intital Noise Frames Count : %d\n",initNoiseFrames);	
		printf(" Noise to Energy Factor : %d\n",thresholdNoiseToEnergyFactor); 
		printf(" Sampling Rate of Recording: %d\n",samplingRate); 
		printf("----------------------------------------------------------------\n\n");	
	}
	else{
		//printing in file
		fprintf(fp_console,"****-------- WELCOME TO HMM --------****\n");		
		fprintf(fp_console,"-Common Settings are : -\n");	
		fprintf(fp_console," P (=Q)(#of Cepstral Coefficients) : %d\n", p);
		fprintf(fp_console," Number of Words/Digits/HMM (W) : %d\n", W);	
		fprintf(fp_console," Number of States per HMM (N) : %d\n", N);	
		fprintf(fp_console," Number of Distinct Observation Symbols (M) or CodeBook Size (Y) : %d\n", M); 	
		fprintf(fp_console," Max Length of Observation Sequence (T) : %d\n", Tmax);			
		fprintf(fp_console," Number of Training Observations : %d\n", On);	
		fprintf(fp_console," Number of Testing Observations : %d\n", Ot);

		fprintf(fp_console,"\n");
		fprintf(fp_console," Frame Size : %d\n", sizeFrame);	
		fprintf(fp_console," Tokhura Weights : ");
		for(int i=0; i<q; i++){
			fprintf(fp_console,"%0.1f(%d) ", w_tkh[i],i+1);
		}
		fprintf(fp_console,"\n Amplitutde Value to Scale : %d\n", scaleAmp);			
		fprintf(fp_console," Intital Header Lines Ignore Count : %d\n", initIgnoreHeaderLines); 
		fprintf(fp_console," Intital Samples to Ignore : %d\n",initIgnoreSamples);	
		fprintf(fp_console," Intital Noise Frames Count : %d\n",initNoiseFrames);	
		fprintf(fp_console," Noise to Energy Factor : %d\n",thresholdNoiseToEnergyFactor); 
		fprintf(fp_console," Sampling Rate of Recording: %d\n",samplingRate); 
		fprintf(fp_console,"----------------------------------------------------------------\n\n");	

	}
}

/**************************************************************************************
	Main Function
**************************************************************************************/
int _tmain(int argc, _TCHAR* argv[])
{

	 /*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
																		 Intialization
	-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	CalculateWeightsForRaisedSineWindow();	// calculating weights for Raised sine window before hand using in program.
	read_codebook_from_file();				// read codebook from file.
	//RecordMyVoice();
	//
	//getch();	return 0;
	char choice;		// choice exercised.	 
  do{
		char ch;
		double accuracy=0, final_accuracy=0;
		unsigned int model_type_use = OutputFolderModel;
		char model_type_string[20] = "new_model";
		bool temp=false;
		char recognised, correct_voice;

		printf("\n\n\n");
		system("pause");
		system("cls");

		printf("\n .............................................. HMM MENU...........................................................");
		//printf("\n u. CodeBook: Generate Cepstral Coeff Universe File From Training Files");
		//printf("\n d. DISPLAY Common System Settings");
		//printf("\n 1. 234101042 : All Training Files");
		//printf("\n 2. 234101042 : All Testing Files");
       // printf("\n 3. CONVERGE: Converge Model Lambda for Each Digit. ");

		printf("\n 1. TESTING: Offline Testing of Digit: Using Old Input Folder Model");

		//printf("\n 5. TESTING: Offline Testing of Digit: Using New Converged Output Folder Model");
		//printf("\n 6. REPLACE: OLD MODEL In Default Input Folder with NEW CONVERGED MODEL in Output Folder.");

		printf("\n 2. TESTING: Live Testing: Using Old Input Folder Model.");
		printf("\n 3. TESTING: Live Testing: Using New Converged Output Folder Model.");

		

		printf("\n 0. Exit - Bye \n Enter your choice : ");
		scanf("%c%*c", &ch);
		
        
		
switch (ch) {
			//case 'd' : DisplayCommonSettings();
				//break;

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											"Training Observation" Sequence Generation for Each Digit
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			//case '1' : 
				// generate observation sequence from the training files. each digit observation sequence is saved in the input lambda folder.
				// also generate normalised files and segrageted data of voice, and analysis of the voice data.
				//sequence_generation(TRAINING);
				//break;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											"Testing Observation" Sequence Generation for Each Digit
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			//case '2' : 
				// generate observation sequence from the testing files. each digit observation sequence is saved in the input lambda folder.
				// also generate normalised files and segrageted data of voice, and analysis of the voice data.
				//sequence_generation(TESTING);
				//break;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											Converge Model Lambda For Each Digit
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			case 'l' : 
					for(int d=0; d<W; d++){
						/**************** Creating necessary file Path for data. ****************/
						sprintf(completePathOuput, "%s%s_%s_HMM_Converged_log.txt", output_folder, WordFolderName, WordNames[d]);   
						/**************** Opening respective files. ****************/
						fp_console = fopen(completePathOuput, "w");					//to read input observation sequence
						if(fp_console == NULL){ 
								perror("\n Error: ");
								printf("\n File Names is: %s \n ", completePathOuput);
								getch();
								return EXIT_FAILURE;
						}
							fprintf(fp_console, "\n  CONVERGING LAMDA : %s %s\n", WordFolderName, WordNames[d]); 	 
								 
						/**************** Reading  Obs Seq from File ****************/
							readObsSeq(d, TRAINING);		// for each digit read their training sequence observations
							fprintf(fp_console, "\n ------------------------------------------------------------------------\n"); 	 

						/**************** Making Converged Lambda From Bakis Model ****************/
							covergence_procedure();				// FOR EACH observation seq generate their model and then converge finally by taking average of all.
							output_lambdaABPi_to_each_file(d);
								printf("\n\t  New Lambda Files Saved: %s%s/%s/", output_folder, output_folder_Model_name, WordNames[d]);
								printf("\n\t  Convergence Done, Log File Generated: %s\n", completePathOuput);
									
								
								fprintf(fp_console, "\n ................................... END ................................................."); 
						 fflush(fp_console); fclose(fp_console); 
					}// for each digit d<W Converge Model Lambda For Each Digit
				break;
			
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											OFFLINE TESTING USING INPUT FOLDER MODEL
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			case '1' : 

				model_type_use = InputFolderModel;
				strcpy(model_type_string,"old_model");
			
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											OFFLINE TESTING USING OUTPUT FOLDER MODEL
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			case '99' : 
				offline_overall_count=0;
				for(int d=0; d<W; d++){
					system("cls");
					/**************** Creating necessary file Path for data. ****************/
					sprintf(completePathOuput, "%s%s_%s_HMM_offline_test_result_%s.txt", output_folder, WordFolderName, WordNames[d],model_type_string);  // {output/}+{WordFolderName}+"_"+{1}+".txt"
					/**************** Opening respective files. ****************/
					fp_console = fopen(completePathOuput, "w");					//to read input observation sequence
					if(fp_console == NULL){ 
							perror("\n Error: ");
							printf("\n File Names is: %s \n ", completePathOuput);
							getch();
							return EXIT_FAILURE;
					}
						fprintf(fp_console, "\n ----------------------- ----------------------- > OFFLINE TESTING : %s %s < ----------------------- -----------------------\n", WordFolderName, WordNames[d]); 	 
						fprintf(fp_console, "\n ------------------------------------------------------------------------\n"); 	 
					/**************** Reading  Obs Seq from File ****************/
						readObsSeq(d, TESTING);
						fprintf(fp_console, "\n ------------------------------------------------------------------------\n"); 	

					/**************** Offline Testing ****************/
						offline_testing(d, model_type_use);

						 accuracy = (double)(offline_correct_count*1.0/Ot)*100;
						printf("\n\t FOR %s %s | Accuracy:  %0.2f %%\n\n", WordFolderName, WordNames[d], accuracy); 
							fprintf(fp_console, "\n\t FOR %s %s | Accuracy:  %0.2f %%\n\n", WordFolderName, WordNames[d], accuracy);

						//printf("\n\n\t -------->> New Lambda Files Saved: %s%s/%s/", output_folder, output_folder_Model_name, WordNames[d]);
						printf("\n\n\t -------->> Offline Testing Done, Log File Generated: %s\n\n", completePathOuput);
							fprintf(fp_console, "\n ---------------------------------- --------------------------------------"); 
							fprintf(fp_console, "\n <---------------------------------- ----------------------------------> END <---------------------------------- -------------------------------------->"); 
					fflush(fp_console); fclose(fp_console); 
					system("pause");
					
				}// for each digit d<W  OFFLINE TESTING
					 final_accuracy = (double)(offline_overall_count*1.0/(Ot*W))*100;
						printf("\n\t Overall Accuracy:  %0.2f %%\n\n", final_accuracy); 
						
				break;	
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
										Replace OLD MODEL In Default Input Folder with NEW CONVERGED MODEL in Output Folder. 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			//case '999' : replace_old_models_files(); 
				//break;
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
										TESTING: Live Testing: Using Old Input Folder Model.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			
			case '2' : model_type_use = InputFolderModel;
					   strcpy(model_type_string,"old_model");
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
										TESTING: Live Testing: Using New Converged Output Folder Model
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
			
			case '3' : 
					printf("\n Duration 3 sec: \n"); 
					timestamp = time(NULL);
					//timestamp = 30;

					/**************** Creating necessary file Path for data. ****************/
					sprintf(liveRecordingFileName, "live_%ld", timestamp);  //file_name
					sprintf(liveRecordingCommand, "%s 3 %s %s%s.txt", recording_module_exe_path, liveRecordingWAV, input_live_voice_folder, liveRecordingFileName);  
					//printf("\n path: %s\n",liveRecordingCommand );
					/**************** Creating necessary file Path for data. ****************/
						sprintf(completePathOuput, "%s%s_test_result_%s.txt", input_live_voice_folder, liveRecordingFileName ,model_type_string);  
						/**************** Opening respective files. ****************/
						fp_console = fopen(completePathOuput, "w");					//to read input observation sequence
						if(fp_console == NULL){ 
								perror("\n Error: ");
								printf("\n File Names is: %s \n ", completePathOuput);
								getch();
								return EXIT_FAILURE;
						}
					// Right Click Project Name in Solution Explorer
					// Select Propertes --> Linker --> Input
					// Select Additional Dependencies --> Edit
					// Add winmm.lib
					do{
						
						
						do
						{
							/**************** Execute the Live Recording Module ****************/
							system(liveRecordingCommand);		//execute the command
							//USAGE : "Recording_Module.exe" <duration_in_seconds> <output_mono_wav_file_path> <output_text_file_path>
						
							printf("\n Playing Sound: "); 
							PlaySound(TEXT(liveRecordingWAV), NULL, SND_SYNC );
							printf("\n\n Is Word Correctly Spoken (y/n) ?" 
								"\n\t n: will repeat the recording process. "
								"\n\n  --Choice :  ");
							//scanf("%c%*c",&correct_voice);
							scanf("%c",&correct_voice);

							while ((getchar()) != '\n');

						}while(correct_voice!='y');
						
						printf("\n\n");

							fprintf(fp_console, "\n ......................... LIVE TESTING : %s ............................ \n",liveRecordingFileName );
						/**************** Live Testing ****************/
							live_testing(model_type_use);

						printf("\n Is Word Correctly Recognised (y/n/e) ?" 
								"\n n: will repeat the recording process again. "
								"\n e: exit. "
								"\n\n  --Choice :  ");

						scanf("%c%*c",&recognised);
						fprintf(fp_console, "\n Digit Correct Recogntion Status (y/n/e): %c", recognised); 
						
						fflush(fp_console);
					 }while(recognised == 'n');
					
							fprintf(fp_console, "\n ......................... END......................................"); 
						fflush(fp_console); fclose(fp_console); 
				break;
			/*case 'u' :   
					//generate_codebook_universe(TRAINING);
						temp=segregate_speech;
						segregate_speech=false;
						codebook_universe_generation=true;
							sequence_generation(TRAINING);
						codebook_universe_generation=false;
						segregate_speech=temp;

				break;*/
			case '0' :   printf("\n Exit \n");  
				break;
			default  :   printf("\n Please enter a valid Choice.\n");
		}//switch
		choice=ch;
	} while (choice != 'n');

	printf("\n....................................ENTER TO EXIT .....................................\n");
	getch();
	return 0;
}

//void generate_codebook_universe(unsigned short int seq_type){
//
//	unsigned short NumOfFiles;
//	char OSeqfilenameIpformatSpecifier[50];
//	char filePathInputVoice[50];
//	char lambda_obs_seq_file_name[50];
//	char seq_type_name[20];
//
//	if(seq_type == 1)
//	{	
//		NumOfFiles = On;
//		strcpy(filePathInputVoice, filePathInputVoiceTraining);
//		strcpy(OSeqfilenameIpformatSpecifier, "training_%s_%s_%s%d");
//		strcpy(lambda_obs_seq_file_name, LambdaFileNames[3]);
//		strcpy(seq_type_name, "TRAINING");
//	}
//	else if(seq_type == 2)
//	{	
//		NumOfFiles = Ot;
//		strcpy(filePathInputVoice, filePathInputVoiceTesting);
//		strcpy(OSeqfilenameIpformatSpecifier, "testing_%s_%s_%s%d");
//		strcpy(lambda_obs_seq_file_name, LambdaFileNames[4]);
//		strcpy(seq_type_name, "TESTING");
//	}
//
//	//to save Cepstral Coefficients 
//	sprintf(OSeqcompletePathFinOp, "%sUniverse.csv", input_folder);  		
//	fp_obsseq_final_op = fopen(OSeqcompletePathFinOp, "w"); 
//	if(fp_obsseq_final_op==NULL){ 
//		perror("\n Error: ");
//		printf("\n File Name : \n  %s\n", OSeqcompletePathFinOp);
//		getch();
//		return ;
//	}
//
//	for(int d = 0 ; d<totDigits ; d++) //iterating through all digits. totDigits
//	{			
//		printf("\n\n\t ---#---#---#---#---#--- GENERATING Cepstral Coefficients of Frames in (%s): %s %s ---#---#---#---#---#---\n", seq_type_name, WordFolderName,  WordNames[d]);
//
//		for(int fileCounter=1; fileCounter <= NumOfFiles ; ++fileCounter)//iterating through all files of given digits (1 to X).
//		{
//		/**************** Creating necessary file Path for data. ****************/
//			
//			//input file name
//			sprintf(OSeqcompletePathIp, "%s%s/%s%d.txt", filePathInputVoice, WordNames[d], voice_data_prefix, fileCounter); 
//			//segregated file data name
//			sprintf(OSeqfileNameIp, OSeqfilenameIpformatSpecifier, WordFolderName, WordNames[d], voice_data_prefix, fileCounter); 
//			sprintf(OSeqcompletePathNorm, "%s%s_normalized_samples.txt", fileOutputRecordingNorSeg, OSeqfileNameIp); 
//			sprintf(OSeqcompletePathNormSeg, "%s%s_normalized_segregated_data.txt", fileOutputRecordingNorSeg, OSeqfileNameIp); 
//			//to save analysis file
//			sprintf(OSeqcompletePathConsole, "%s%s_analysis.txt", fileOutputRecordingAnalysis, OSeqfileNameIp);  
//			/**************** Opening respective files. ****************/
//			fp_obs_seq_ip = fopen(OSeqcompletePathIp, "r");				//to read input file
//			fp_obs_seq_norm = fopen(OSeqcompletePathNorm, "w+");		//to save normalised samples
//			fp_obsseq_norm_seg = fopen(OSeqcompletePathNormSeg, "w");  //to save segregated recording from start to end
//			fp_obsseq_console = fopen(OSeqcompletePathConsole, "w");	// to save analysis data of each file
//			if(fileCounter==1){
//				DisplayCommonSettingsObsSeq(fp_obsseq_console);
//			}
//			if(fp_obs_seq_ip == NULL || fp_obs_seq_norm == NULL || fp_obsseq_norm_seg == NULL ||  fp_obsseq_console==NULL ){ 
//					perror("\n Error: ");
//					printf("\n File Names are : \n  %s, \n  %s, \n  %s, \n %s \n", OSeqcompletePathIp, OSeqcompletePathNorm, OSeqcompletePathNormSeg, OSeqcompletePathConsole  );
//					getch();
//					return ;
//			}
//			
//		if(fileCounter==1){
//			printf("  ----> FILE: %s,\n", OSeqcompletePathIp);  
//		}
//		else
//		{
//			printf("\t %s%d.txt,", voice_data_prefix, fileCounter);  
//		}
//
//
//		fprintf(fp_obsseq_console, "\n ----------------------- START - ANALYZING OF FILE: %s ----------------------- \n", OSeqcompletePathIp);
//
//		/**************** DC Shift and Normalizing ****************/
//			normalize_dcshift_samples();
//
//		/**************** Frames ZCR and Energy. STE Marker ****************/
//			zcr_energy_frames();
//
//		   //if(segregate_speech){						//only if you want to segregate speech into separate file.
//			/****************  calculating noise energy and threshold. ****************/
//				noiseEnergy_thresholds_frames();						// if you want to calculate thresholds for zcr and energy
//					
//			/**************** start and end marker of speech ****************/
//				marker_start_end_segregated();							//this and above func, if you want to detect start, end marker of speech, and to save it in separate file.
//				fclose(fp_obsseq_norm_seg);	// closing file stream
//			//}
//		   //else
//		   //{
//			  // fclose(fp_obsseq_norm_seg);		// closing file stream
//			  // remove(OSeqcompletePathNormSeg);		//removing unnecessory file created.
//		   //}
//			if(!segregate_speech)
//			{
//				remove(OSeqcompletePathNormSeg);		//removing unnecessory
//			}
//
//		  // closing file stream, as no longer needed.
//		   fflush(fp_obs_seq_ip); fclose(fp_obs_seq_ip); 
//		   fflush(fp_obs_seq_norm); fclose(fp_obs_seq_norm);
//		   remove(OSeqcompletePathNorm);	//comment it if you want to keep normalised data file.
//
//		/****************  Calculating Coefficients for Voiced Frames of File ****************/
//			long totFramesVoice = end-start+1;
//			calculateCoefficientsForFramesOfSpeech(totFramesVoice); //for each frame calculate coefficients
//
//			for(int ff=0; ff<totFramesVoice; ff++){
//				for(int i=1;i<=p;i++){
//					fprintf(fp_obsseq_final_op, "%lf,", C_rsw[ff][i]);
//				}
//				fprintf(fp_obsseq_final_op, "\n");
//			}
//				
//				//printf("\n ----------------------- END Analyzing OF File: %s ----------------------- \n", OSeqfileNameIp);  
//				fprintf(fp_obsseq_console, "\n ----------------------- END - ANALYZING OF FILE: %s ----------------------- \n", OSeqfileNameIp);
//		
//			fflush(fp_obsseq_console); fclose(fp_obsseq_console);
//		}//end of filecounter loop -------------------------------------------------------------------------------------------------------------------
//		//system("pause");
//	}//end of digit loop ------------------------------------------------------------------------------------------------------------------------------
//		
//	printf("\n\n  ----> CodeBook Universe File Generated: %s\n\n", OSeqcompletePathFinOp); 
//	printf("\n-----------------------------------------------------\n");
//	fflush(fp_obsseq_final_op); fclose(fp_obsseq_final_op);
//}//generate_codebook_universe