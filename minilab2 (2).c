#include <stdio.h>
#include "L138_LCDK_aic3106_init.h"
#include "L138_LCDK_switch_led.h"
#include <ti/dsplib/dsplib.h>
#include "evmomapl138_gpio.h"
#include <stdint.h>
#include <math.h>


// Global Definitions and Variables
#define PI 3.14159265358979323
#define N 1024


int idx = 0;

/* Align the tables that we have to use */

// The DATA_ALIGN pragma aligns the symbol in C, or the next symbol declared in C++, to an alignment boundary.
// The alignment boundary is the maximum of the symbol's default alignment value or the value of the constant in bytes.
// The constant must be a power of 2. The maximum alignment is 32768.
// The DATA_ALIGN pragma cannot be used to reduce an object's natural alignment.

//The following code will locate mybyte at an even address.
//#pragma DATA_ALIGN(mybyte, 2)
//char mybyte;

//The following code will locate mybuffer at an address that is evenly divisible by 1024.
//#pragma DATA_ALIGN(mybuffer, 1024)
//char mybuffer[256];
#pragma DATA_ALIGN(x_in,8);
int16_t x_in[2*N];

#pragma DATA_ALIGN(x_sp,8);
float   x_sp [2*N];
#pragma DATA_ALIGN(y_sp,8);
float   y_sp [2*N];
#pragma DATA_ALIGN(w_sp,8);
float   w_sp [2*N];

int16_t sound_array[1024];
int16_t windowed_sound_array[1024];
int index = 0;
int cycle = 0;
float audio_mel [28][2];
int lambda_n[28];
int audio_mel_index=0;
int flag1 = 0;
int flag2 = 0;

float Y[13] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

/// NEURAL NETWORK COEFFICIENTS ////
float x1_step1_xoffset[13] = {17.165012,-5.755931,11.810664,-5.260928,4.630504,-0.839859,5.797121,-0.095937,4.50044,-1.899734,6.735817,-1.226769,6.438809};
float x1_step1_gain[13] = {0.0724655351197513,0.117613981620816,0.132995232852376,0.0916207299414392,0.133589620567405,0.230510748428061,0.153804177136886,0.179487117357023,0.163548973472602,0.247062641373876,0.264528045594054,0.288150557513699,0.248771968275106};
float x1_step1_ymin = -1.0;
// layer 1
float b1[5] = {-1.0022091002765931567,-0.80969421549275721883,0.3780511575442287997,-0.86859534902379997856,-0.51277539346265399445};
float IW1_1[5][13] = {{1.1104577096233236855, 0.58062671066840043643, 0.19394925404966761873, -0.53691173622602850202, -0.47186817131207570153, -0.69995905923905632484, 0.74132436049112437892, -1.1019773268779688991, -0.20000725341919611822, 0.057531287889454968409, -0.43416492853540827879, -0.065901764275768312529, -1.063132485732153798},
		{0.8628266446154224667, -0.30969057259325627474, -0.42776295614304499226, 0.85822806931250628093, 0.88867977848996670964, -0.28559447861877579333, -0.71464799379117471823, -0.73058091266495539529, 0.19000847462820394385, 0.70429007839581969641, 0.34196955045427462894, -0.68436778782864438053, 0.16565088914453682256},
		{1.2590745177432103308, 0.90302689448826789498, -1.3926753478295088584, -0.83927742190864373928, 0.57063000655860540711, 0.085887043166967200203, 0.53286070609424696087, -0.70830508432190153467, 0.028128730576524534068, 0.048501173535170916118, 0.11486384306376958009, 0.25785508325450046119, -1.058517531674779022},
		{1.2147996914850347494, -0.48514739868933209888, 0.17183729855491455818, 0.78907628841153898414, 0.66146355095454878459, 0.39579918753229814676, -0.48240661555490482737, -1.4545964540238036644, -0.41562451560900104397, 0.34706574483837004941, -1.0311882947980077763, -0.38002063428228605169, -0.21034124828904648963},
		{0.2628166170138283575, 0.63211660072992703618, 0.13914513043185361418, -0.96350521109866771319, 8.6345385445971839511e-05, -0.33785933117520300373, -0.73777276304039363097, 1.0316565440214233718, 0.3394219802286199128, -1.080409462209888094, 0.542660348905921186, 0.65753509032547952096, -0.66606180028998018816}};
float h1[5]={0.0,0.0,0.0,0.0,0.0};
float S1[5]={0.0,0.0,0.0,0.0,0.0};
// layer 2
float h2[4];
float b2[4] = {-0.04565895460758535862,-0.28035148256599484728,0.59660162818814077568,-0.48832047173975540177};
float LW2_1[4][5] = {{1.8237494473273656581, -1.2492996401992140232, 1.3488487918948033339, -0.27536875049669273796, 0.7884337609130397384},
		{-0.92378848452500861299, -0.60330802864658583662, 1.0702503989280471863, -2.0268450473399899359, 1.5414614153892440829},
		{0.45714437905857058242, 1.2428704526917015993, 0.78239022845310468579, 2.333946261929771282, -0.18420811325572125638},
		{-0.27313175597493344338, -0.63323532947661376191, -2.3245501030166075829, -1.0000057742027848029, -1.0549985905678513909}};

float sum_Y1[5]={0.0,0.0,0.0,0.0,0.0};
float sum_Y2[5]={0.0,0.0,0.0,0.0,0.0};
float sum_exp=0.0;
float output[4]={0.0,0.0,0.0,0.0};


/// END NN COEFFFSS?///////




// brev routine called by FFT routine
unsigned char brev[64] = {
    0x0, 0x20, 0x10, 0x30, 0x8, 0x28, 0x18, 0x38,
    0x4, 0x24, 0x14, 0x34, 0xc, 0x2c, 0x1c, 0x3c,
    0x2, 0x22, 0x12, 0x32, 0xa, 0x2a, 0x1a, 0x3a,
    0x6, 0x26, 0x16, 0x36, 0xe, 0x2e, 0x1e, 0x3e,
    0x1, 0x21, 0x11, 0x31, 0x9, 0x29, 0x19, 0x39,
    0x5, 0x25, 0x15, 0x35, 0xd, 0x2d, 0x1d, 0x3d,
    0x3, 0x23, 0x13, 0x33, 0xb, 0x2b, 0x1b, 0x3b,
    0x7, 0x27, 0x17, 0x37, 0xf, 0x2f, 0x1f, 0x3f
};

// The seperateRealImg function separates the real and imaginary data
// of the FFT output. This is needed so that the data can be plotted
// using the CCS graph feature
float y_real_sp [N];
float y_imag_sp [N];
float y_mag_firstK [513];
float filter_bank [26][513];
float triangleEnergy [26] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}; // energy in each triangle...
float logTriangleEnergy [26]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
float Z_MFCC[13] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

separateRealImg () {
    int i, j;

    for (i = 0, j = 0; j < N; i+=2, j++) {
        y_real_sp[j] = y_sp[i];
        y_imag_sp[j] = y_sp[i + 1];
    }
}

// Function for generating sequence of twiddle factors
void gen_twiddle_fft_sp (float *w, int n)
{
    int i, j, k;
    double x_t, y_t, theta1, theta2, theta3;

    for (j = 1, k = 0; j <= n >> 2; j = j << 2)
    {
        for (i = 0; i < n >> 2; i += j)
        {
            theta1 = 2 * PI * i / n;
            x_t = cos (theta1);
            y_t = sin (theta1);
            w[k] = (float) x_t;
            w[k + 1] = (float) y_t;

            theta2 = 4 * PI * i / n;
            x_t = cos (theta2);
            y_t = sin (theta2);
            w[k + 2] = (float) x_t;
            w[k + 3] = (float) y_t;

            theta3 = 6 * PI * i / n;
            x_t = cos (theta3);
            y_t = sin (theta3);
            w[k + 4] = (float) x_t;
            w[k + 5] = (float) y_t;
            k += 6;
        }
    }
}

interrupt void interrupt4(void) // interrupt service routine
{
  int16_t left_sample;
  left_sample = input_left_sample();

  	  LCDK_LED_off(5);
  	  LCDK_LED_off(6);
  	  LCDK_LED_off(7);
  	  LCDK_LED_off(4);

  if(LCDK_SWITCH_state(5)==1){
	  LCDK_LED_on(5);
	  LCDK_LED_off(6);
	  LCDK_LED_off(7);
	  LCDK_LED_off(4);
	  // index ranges from 0 to 1024 while cycle ranges from 16000 to 1024 + 16000
	  if (cycle >= 16000 && cycle < (1024+16000)){
		  	  LCDK_LED_on(5);
		 	  LCDK_LED_on(6);
		 	  LCDK_LED_on(7);
		 	  LCDK_LED_on(4);

		  sound_array[index]=left_sample;
		  windowed_sound_array[index] = sound_array[index] * (0.54-(0.56*cos((2.0*PI*index)/1023))) ;

		  // Input is being read sample by sample real part in even indices, imaginary in odd.
		  x_in[2*index]=windowed_sound_array[index];
		  x_in[2*index+1]=(float)0.0;

		  index++;

	  }

		  // cycle increments regardless
	  	  if (cycle <= (1024+16000)){
		  cycle = cycle+1;
	  	  }
	  	  if(cycle == (1024+16000)){
	  		  flag1 = 1;
	  	  }

  }

  else {
	  	  LCDK_LED_off(5);
	  	  LCDK_LED_off(6);
	  	  LCDK_LED_off(7);
	  	  LCDK_LED_off(4);
  }



  // Output to DAC (Line OUT)
	output_left_sample(left_sample);
	return;
}



int main(void)
{
  L138_initialise_intr(FS_16000_HZ,ADC_GAIN_0DB,DAC_ATTEN_0DB,LCDK_MIC_INPUT);
  LCDK_LED_init();
  LCDK_SWITCH_init();
  // SAMPLE CODE: USE OF FFT ROUTINES

	// Copy input data to the array used by DSPLib functions
  int n;
	for (n=0; n<N; n++)
	{
	  x_sp[2*n]   = x_in[2*n];
	  x_sp[2*n+1] = x_in[2*n+1];
	}

	// Call twiddle function to generate twiddle factors needed for FFT and IFFT functions
  gen_twiddle_fft_sp(w_sp,N);

  // Call FFT routine
  DSPF_sp_fftSPxSP(N,x_sp,w_sp,y_sp,brev,4,0,N);

  // Call routine to separate the real and imaginary parts of data
  // Results saved to floats y_real_sp and y_imag_sp
  separateRealImg ();



  int i = 0;
  int m = 0;
  int k = 0;
  for (i = 0; i < 513; i++){
	  y_mag_firstK[i] = y_real_sp[i]*y_real_sp[i] + y_imag_sp[i]*y_imag_sp[i]; // first K points of SQUARED MAGNITUDE
  }

  // computing the mel frequencies and audio frequency range
  audio_mel[0][0]= 250;
  audio_mel[27][0]= 8000;
  audio_mel[0][1]= 344.16;
  audio_mel[27][1]= 2840.02;

  for(i = 1; i<27; i++)
  {
	  audio_mel[i][1] = audio_mel[i-1][1]+92.44666;
	  audio_mel[i][0] = 700*(pow(10, audio_mel[i][1]/2595)-1);
  }

  // creating the lambda bank
  for(i=0; i<28 ;i++){
	 // we want audio_mel[i][0] to be audio frequency equivalent to the ith mel frequency
  	 lambda_n[i] =  (int) audio_mel[i][0]*(0.064125);
  }

  // populate filter bank
  for (m = 1; m < 27; m++){ // i is filter index (same as m in notes)
	  for (n = 1; n <= 513; n++){ // n is frequency index (same as k in notes)
		  if (n < lambda_n[m-1]){
			  filter_bank[m-1][n-1]=0;
		  }
		  if (n >= lambda_n[m-1] && n <= lambda_n[m]){
			  filter_bank[m-1][n-1] = ((float)(n-lambda_n[m-1]))/((float)(lambda_n[m]-lambda_n[m-1]));
		  }

		  if (n <= lambda_n[m+1] && n >= lambda_n[m]){
		  			  filter_bank[m-1][n-1] = ((float)(lambda_n[m+1]-n))/((float)(lambda_n[m+1]-lambda_n[m]));
		  }

		  if (n > lambda_n[m+1]){
		  			  filter_bank[m-1][n-1]=0;
		  }
	  }
  }

//memset(triangleEnergy, 0, 26);

// for each triangle...
for (m = 0; m < 27; m++){
	for (i = 0; i < 513; i++){ // for each element in y_mag_firstK
		triangleEnergy[m] += y_mag_firstK[i] * filter_bank[m][i];
	}
}

// log the triangle energy
for (m = 0; m < 27; m++){
	logTriangleEnergy[m] = log10(triangleEnergy[m]); // logTriangleEnergy is X_m in the lab
}

//Discrete Cosine Transform of Xm to compute ZMFCC values

for(i=1; i<=13; i++){ // i's in instructions
	for(k=0; k<26; k++){ // m's in instructions

		Z_MFCC[i-1] += logTriangleEnergy[k] * cos(i*(k-0.5)*(PI/26.0));

	}
	/*if (i == 13){
		flag2 = 1;
	}*/
}


//LAYER 1!!
// PreProcessing
for(i=0; i<13 ; i++){
    // y is a 13-vector
	//subtraction
	Y[i] = Z_MFCC[i] - x1_step1_xoffset[i];

	//multiplication
	Y[i] = Y[i] * x1_step1_gain[i];

	//Addition
	Y[i]= Y[i] + x1_step1_ymin;
}
// take the sum for layer 1
// IW_1 is a 5x13 matrix mxn
// sum_Y1 is a 5-vector
for (m=0; m< 5; m++){
	for(i=0; i < 13; i++){
		sum_Y1[m] += IW1_1[m][i] * Y[i];
	}
}
// add the biases for layer 1
for (m=0; m<5; m++){
	S1[m]=sum_Y1[m]+b1[m];
}

// normalize for layer 1. This gives layer 2's inputs
for (m=0; m<5; m++){
	h1[m] = (2.0/(1+exp(-2.0*S1[m])))-1.0;
}

// LAYER 2
// take the sum for layer 2
for (m=0; m< 4; m++){
	for(i=0; i < 5; i++){
		sum_Y2[m] += LW2_1[m][i] * h1[i];
	}
}
// add the biases for layer 2
for (m=0; m<4; m++){
	h2[m]=sum_Y2[m]+b2[m];
}

// Softmax to output
for (m=0; m<4; m++){
	sum_exp += exp(h2[m]);
}

for (m=0; m<4; m++){
	output[m] = exp(h2[m])/sum_exp;
}
// END LAYER 2


  // Call the inverse FFT routine
  DSPF_sp_ifftSPxSP(N,y_sp,w_sp,x_sp,brev,4,0,N);

  // END OF SAMPLE CODE
while (1);

}











