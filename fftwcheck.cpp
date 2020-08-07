/*

nvcc fftwcheck.cpp  -lcufft  -o myCufftApp `pkg-config fftw3 --libs` `pkg-config opencv4 --cflags` `pkg-config opencv4 --libs`

*/
 
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <math.h>
#include <iostream>
#include <vector>
#define _USE_MATH_DEFINES

using namespace std;

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <cuda_runtime.h>
#include <cufft.h>
#include "fftw3.h"
#pragma comment( lib, "fftw3.lib" )

 
#define SIZEX 256
#define SIZEY 256
#define SIZEZ 1
#define SIZE (SIZEX*SIZEY*SIZEZ)


cv::Mat_<uchar> image = cv::imread("a.jpg", 0);

float fft_body3dR[SIZE];
float fft_body3dI[SIZE];

int DCexchange2D(float* dataR, float* dataI, int cols, int rows, int depth )
{
	int i,j,k;
	int p1,p2,p3;    // point position
	int c2,r2,d2;    // temporary for cols/2,rows/2
	float re,im; // temporary

	//if( data==NULL )       return false;
	if( (rows<0 || cols<0) || depth<0) return false;
 
	c2 = cols/2;
	r2 = rows/2;
    d2 = depth/2;
 
	for( k=0; k<depth; k++ ){
		for( j=0; j<r2; j++ ){
            for ( i=0; i<cols; i++ ){
                // exchange p1( i, j ) <-> p2( (cols/2+i)%cols, rows/2+j )
                p1 = j*cols + i + k*rows*cols;
                p2 = (r2+j)*cols + (c2+i)%cols + k*rows*cols;
                re = dataR[p1];
                im = dataI[p1];
                dataR[p1] = dataR[p2];
                dataI[p1] = dataI[p2];
                dataR[p2] = re;
                dataI[p2] = im;
            }
		}
	}

	for( k=0; k<d2; k++ ){
		for( j=0; j<rows; j++ ){
            for ( i=0; i<cols; i++ ){
                // exchange p1( i, j ) <-> p2( (cols/2+i)%cols, rows/2+j )
                p1 = j*cols + i + k*rows*cols;
                p2 = j*cols + i + (d2+k)*rows*cols;
                re = dataR[p1];
                im = dataI[p1];
                dataR[p1] = dataR[p2];
                dataI[p1] = dataI[p2];
                dataR[p2] = re;
                dataI[p2] = im;
            }
		}
	}

	return true;
}

int fft3d(void){


	cufftComplex *cuin_h  = NULL;
	cufftComplex *cuout_h  = NULL;
	cufftComplex *cuin  = NULL;
	cufftComplex *cuout = NULL;
	cufftComplex *cuout2 = NULL;
	cufftHandle cup, cuip;
	int i,j,k,idx;

	/*
	fftw_complex *in  = NULL;
	fftw_complex *out = NULL;
	fftw_complex *out2 = NULL;
	fftw_plan p, ip   = NULL;
	int i,j,k,idx;
	*/
 
//************************************************************************************

	cout << "hi" << endl;

	size_t cumem_size = sizeof(cufftComplex) * SIZE;
	cuin_h = (cufftComplex*) malloc( cumem_size);
	cuout_h = (cufftComplex*) malloc( cumem_size);
	cudaMalloc((void**)&cuin,cumem_size);
	cudaMalloc((void**)&cuout,cumem_size);
	cudaMalloc((void**)&cuout2,cumem_size);

	/*
	size_t mem_size = sizeof(fftw_complex) * SIZE;
	in  = (fftw_complex*)fftw_malloc( mem_size );
	out = (fftw_complex*)fftw_malloc( mem_size );
	out2 = (fftw_complex*)fftw_malloc( mem_size );

	if( !in || !out ){
		fprintf( stderr, "failed to allocate %d[byte] memory(-.-)\n", (int)mem_size );
		return false;
	}
	*/


//************************************************************************************

	cufftPlan3d(&cup, SIZEY, SIZEX, SIZEZ, CUFFT_C2C);
	cufftPlan3d(&cuip, SIZEY, SIZEX, SIZEZ, CUFFT_C2C);
 
	/*
	// !! row-major alignment is recommended, but here, column-major.
	p = fftw_plan_dft_3d( SIZEY, SIZEX, SIZEZ, in, out, FFTW_FORWARD, FFTW_ESTIMATE );
	ip = fftw_plan_dft_3d( SIZEY, SIZEX, SIZEZ, out, out2, FFTW_BACKWARD, FFTW_ESTIMATE );
	*/

//************************************************************************************




	// input data creation
	//printf("----- INPUT -----\n");
	for( k=0; k<SIZEZ; k++ ){
        for( j=0; j<SIZEY; j++ ){
            for( i=0; i<SIZEX; i++ ){
                idx = SIZEZ*k+SIZEX*j+i; // column-major alignment
                //in[idx][0] = image[i][j];  //1 + 2*sin(2*M_PI*i/SIZEX) + sin(4*M_PI*j/SIZEY);
				cuin_h[idx] = make_cuComplex(image[i][j],image[i][j]);
                //in[idx][1] = 0;
            }
        }
    }
	cudaMemcpy( cuin, cuin_h, cumem_size, cudaMemcpyHostToDevice);
 


//************************************************************************************

	cufftExecC2C(cup, cuin, cuout, CUFFT_FORWARD);

	//fftw_execute(p);



//************************************************************************************

	cudaMemcpy( cuout_h, cuout, cumem_size,cudaMemcpyDeviceToHost);


	/*for( i=0; i<SIZE; i++ ){
		fft_body3dR[i] = cuCrealf(cuout_h[i]);
		fft_body3dI[i] = cuCimagf(cuout_h[i]);
	}*/
	for( j=0; j<SIZEY; j++ ){
		for( i=0; i<SIZEX; i++ ){
			idx = SIZEX*j+i;
			//printf("%d %d %lf %lf\n", i, j, out[idx][0]*scale, out[idx][1]*scale );
            //image[i][j] = int(abs(out[idx][0]*scale*255));
			//image[i][j] = int(abs(cuCrealf(cuout_h[idx]))*scale*255);
			fft_body3dR[idx] = cuCrealf(cuout_h[idx]);
			fft_body3dI[idx] = cuCimagf(cuout_h[idx]);
		}
	}
    DCexchange2D(fft_body3dR, fft_body3dI, SIZEX, SIZEY, SIZEZ);
    //cv::Mat resultimage[SIZEX][SIZEY];
 
	// output is DC exchanged and scaled.
	double scale = 1. / SIZE;
	//printf("\n----- RESULT -----\n");
	for( j=0; j<SIZEY; j++ ){
		for( i=0; i<SIZEX; i++ ){
			idx = SIZEX*j+i;
			//printf("%d %d %lf %lf\n", i, j, out[idx][0]*scale, out[idx][1]*scale );
            //image[i][j] = int(abs(out[idx][0]*scale*255));
			//image[i][j] = int(abs(cuCrealf(cuout_h[idx]))*scale*255);
			image[i][j] = int(abs(fft_body3dR[idx])*scale*255);
		}
	}


    cv::imwrite("cu_Result1.png", image);

	cufftExecC2C(cup, cuout, cuout2, CUFFT_INVERSE);
	cudaMemcpy( cuout_h, cuout2, cumem_size,cudaMemcpyDeviceToHost);

    //DCexchange2D(out2, SIZEX, SIZEY, SIZEZ);
	//fftw_execute(ip);


	//printf("\n----- RESULT -----\n");
	for( j=0; j<SIZEY; j++ ){
		for( i=0; i<SIZEX; i++ ){
			idx = SIZEX*j+i;
			//printf("%d %d %lf %lf\n", i, j, out2[idx][0]*scale, out2[idx][1]*scale );
            //image[i][j] = int(abs(out2[idx][0]*scale));
			//image_instead[idx] = cuCrealf(cuout_h[idx]);
			image[i][j] = int(abs(cuCrealf(cuout_h[idx]))*scale);
		}
	}
    cv::imwrite("cu_Result2.png", image);


	cout << "hihi" << endl;

//************************************************************************************

	if( cuip   ) cufftDestroy(cuip);
	if( cup   ) cufftDestroy(cup);
	if( cuin  ) cudaFree(cuin);
	if( cuout ) cudaFree(cuout);
	if( cuout2 ) cudaFree(cuout2);

	/*
	if( ip   ) fftw_destroy_plan(ip);
	if( p   ) fftw_destroy_plan(p);
	if( in  ) fftw_free(in);
	if( out ) fftw_free(out);
	if( out2 ) fftw_free(out2);
	*/

    return true;
}



int main( void ){

    fft3d();
    //vector<cv::Mat> planes;
    //cv::split(resultimage, planes);

    //cout << "dims:" << resultimage.dims << endl;
    //cv::imwrite("Result.png", image);
 
	return true;
}
