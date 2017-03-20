#include <mmintrin.h> /* MMX */
#include <xmmintrin.h> /* SSE, нужен также mmintrin.h */
#include <emmintrin.h> /* SSE2, нужен также xmmintrin.h */
#include <pmmintrin.h> /* SSE3, нужен также emmintrin.h */
#include <smmintrin.h> /* SSE4.1 */
#include <nmmintrin.h> /* SSE4.2 */
#include <immintrin.h> /* AVX */
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;
Mat img,image;

int pixelMatrix[3][3] = { 0 };
//float GY[6] = { -1,-2,-1,1,2,1 };
//float GX[6] = { 1,-1,2,-2,1,-1 };
float GY[6] = { -1,-2,-1,0,2,1 };
float GX[6] = { 0,-1,2,-2,0,-1 };
double GY_d[6] = { -1,-2,-1,1,2,1 };
double GX_d[6] = { 1,-1,2,-2,1,-1 };

int open();
int save();
void show();
void sobel();
double convolution();
double add_sse(float *m_gy, float *m_gx);
double add_sse2(double *m_gy, double *m_gx);

int main(){
	open();	
	sobel();
	save();
	//show();
}
int open() {
	img = imread("D:/1.jpg", CV_LOAD_IMAGE_COLOR); //read the image data in the file "MyPic.JPG" and store it in 'img'

	if (img.empty()) {
		cout << "Error : Image cannot be loaded..!!" << endl;
		//system("pause"); //wait for a key press
		return -1;
	}
	
}
int save() {
	bool bSuccess = imwrite("D:/TestImage.jpg", image); //write the image to file

	if (!bSuccess) {

		cout << "ERROR : Failed to save the image" << endl;

		//system("pause"); //wait for a key press
		return -1;
	}
	
}
void show() {
	namedWindow("MyWindow", CV_WINDOW_NORMAL); //create a window with the name "MyWindow"
	imshow("MyWindow", image); //display the image which is stored in the 'img' in the "MyWindow" window

	waitKey(0); //wait infinite time for a keypress

	destroyWindow("MyWindow"); //destroy the window with the name, "MyWindow"
}
void sobel() {
	int **a = new int *[20000];
	for (int i = 0; i != 20000; ++i){
		a[i] = new int[20000];
	}
	image = img;
	for (int i = 0; i<img.cols; i++){
		for (int j = 0; j<img.rows; j++){
			// get pixel
			Vec3b color = image.at<Vec3b>(Point(i, j));
			int red = color.val[2];
			a[i][j] = red;
		}
	}
	for (int i = 1; i < img.cols - 1; i++) {
		for (int j = 1; j < img.rows - 1; j++) {
			pixelMatrix[0][0] = a[i - 1][j - 1];
			pixelMatrix[0][1] = a[i - 1][j];
			pixelMatrix[0][2] = a[i - 1][j + 1];


			pixelMatrix[1][0] = a[i][j - 1];
			pixelMatrix[1][2] = a[i][j + 1];


			pixelMatrix[2][0] = a[i + 1][j - 1];
			pixelMatrix[2][1] = a[i + 1][j];
			pixelMatrix[2][2] = a[i + 1][j + 1];
			
			float gy[6] = { (float)pixelMatrix[0][0],(float)pixelMatrix[0][1],(float)pixelMatrix[0][2],(float)pixelMatrix[2][0],
				(float)pixelMatrix[2][1],(float)pixelMatrix[2][2] };
			float gx[6] = { (float)pixelMatrix[0][0],(float)pixelMatrix[0][2],(float)pixelMatrix[1][0], (float)pixelMatrix[1][2],
				(float)pixelMatrix[2][0],(float)pixelMatrix[2][2] };

			Vec3b color = image.at<Vec3b>(Point(i, j));
			//int edge = (int)add_sse(gy,gx);
			int edge = (int)convolution();
			color.val[0] = edge;
			color.val[1] = edge;
			color.val[2] = edge;
			
			// set pixel 
			image.at<Vec3b>(Point(i, j)) = color; 
		}
	}
}
double convolution() {

	int gy = (pixelMatrix[0][0]=pixelMatrix[0][0] * -1) + (pixelMatrix[0][1]= pixelMatrix[0][1] * -2) + (pixelMatrix[0][2]=pixelMatrix[0][2] * -1) + (pixelMatrix[2][0]) + (pixelMatrix[2][1]=pixelMatrix[2][1] * 2) + (pixelMatrix[2][2]=pixelMatrix[2][2] * 1);
	//cout << gy << "\n";
	int gx = (pixelMatrix[0][0]) + (pixelMatrix[0][2] =pixelMatrix[0][2] * -1) + (pixelMatrix[1][0]=pixelMatrix[1][0] * 2) + (pixelMatrix[1][2]=pixelMatrix[1][2] * -2) + (pixelMatrix[2][0]) + (pixelMatrix[2][2]=pixelMatrix[2][2] * -1);
	//cout << gx << "\n";
	//cout << "\n";
	/*for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			cout << pixelMatrix[i][j] << " ";
		}
		cout << "\n";
	}*/
	return sqrt(pow(gy, 2) + pow(gx, 2));
}
double add_sse(float *m_gy, float *m_gx){ //SSE
	__m128 t0,t0_1, t1,t1_1;
	t0 = _mm_load_ps(m_gy);
	t0_1 = _mm_load_ps(GY);

	t1 = _mm_load_ps(m_gx);
	t1_1 = _mm_load_ps(GX);

	t0 = _mm_mul_ps(t0, t0_1);
	t1 = _mm_mul_ps(t1, t1_1);

	_mm_store_ps(m_gy, t0_1);
	_mm_store_ps(m_gx, t1_1);

	float sum_gy = 0.0;
	for (int i = 0; i < 6; i++){
		sum_gy = sum_gy + m_gy[i];		
	}

	float sum_gx = 0.0;
	for (int i = 0; i < 6; i++) {
		sum_gx = sum_gx + m_gx[i];
	}

	return sqrt(pow(sum_gy, 2) + pow(sum_gx, 2));

}
double add_sse2(double *m_gy, double *m_gx) { //SSE2
	__m128d t0, t0_1, t1, t1_1;
	t0 = _mm_load_pd(m_gy);
	t0_1 = _mm_load_pd(GY_d);

	t1 = _mm_load_pd(m_gx);
	t1_1 = _mm_load_pd(GX_d);

	t0 = _mm_mul_pd(t0, t0_1);
	t1 = _mm_mul_pd(t1, t1_1);

	_mm_store_pd(m_gy, t0);
	_mm_store_pd(m_gx, t1);

	float sum_gy = 0;
	for (int i = 0; i < 6; i++) {
		sum_gy = sum_gy + m_gy[i];
	}
	float sum_gx = 0;
	for (int i = 0; i < 6; i++) {
		sum_gx = sum_gx + m_gx[i];
	}
	return sqrt(pow(sum_gy, 2) + pow(sum_gx, 2));

}
double add_sse4(int *m_gy, int *m_gx) { //SSE 4.1


}