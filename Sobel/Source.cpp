#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;
Mat img,image;

int pixelMatrix[3][3] = { 0 };

int open();
int save();
void show();
void sobel();
double convolution();

int main(){
	open();	
	sobel();
	save();
	show();
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
	int **a = new int *[9000];
	for (int i = 0; i != 9000; ++i){
		a[i] = new int[9000];
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

			Vec3b color = image.at<Vec3b>(Point(i, j));
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

	int gy = (pixelMatrix[0][0] * -1) + (pixelMatrix[0][1] * -2) + (pixelMatrix[0][2] * -1) + (pixelMatrix[2][0]) + (pixelMatrix[2][1] * 2) + (pixelMatrix[2][2] * 1);
	int gx = (pixelMatrix[0][0]) + (pixelMatrix[0][2] * -1) + (pixelMatrix[1][0] * 2) + (pixelMatrix[1][2] * -2) + (pixelMatrix[2][0]) + (pixelMatrix[2][2] * -1);
	return sqrt(pow(gy, 2) + pow(gx, 2));
	//
}