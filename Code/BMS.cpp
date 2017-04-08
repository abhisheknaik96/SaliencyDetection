/*****************************************************************************
*	Implemetation of the saliency detction method described in paper
*	"Exploit Surroundedness for Saliency Detection: A Boolean Map Approach",
*   Jianming Zhang, Stan Sclaroff, submitted to PAMI, 2014
*
*	Copyright (C) 2014 Jianming Zhang
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*	If you have problems about this software, please contact: jmzhang@bu.edu
*******************************************************************************/

#include <fstream>
#include <iostream>

#include <vector>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstring>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "fileGettor.h"

using namespace cv;
using namespace std;

#define MAX_IMG_DIM 400
#define COV_MAT_REG 50.0f

static const int CL_RGB = 1;
static const int CL_Lab = 2;
static const int CL_Luv = 4;

static cv::RNG BMS_RNG;

class BMS
{
	public:
		BMS (const cv::Mat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening);
		cv::Mat getSaliencyMap();
		void computeSaliency(double step);
	private:
		cv::Mat mSaliencyMap;
		int mAttMapCount;
		cv::Mat mBorderPriorMap;
		cv::Mat mSrc;
		std::vector<cv::Mat> mFeatureMaps;
		int mDilationWidth_1;
		bool mHandleBorder;
		bool mNormalize;
		bool mWhitening;
		int mColorSpace;
		cv::Mat getAttentionMap(const cv::Mat& bm, int dilation_width_1, bool toNormalize, bool handle_border);
		void whitenFeatMap(const cv::Mat& img, float reg);
		void computeBorderPriorMap(float reg, float marginRatio);
};

void postProcessByRec8u(cv::Mat& salmap, int kernelWidth);
void postProcessByRec(cv::Mat& salmap, int kernelWidth);



BMS::BMS(const Mat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening)
:mDilationWidth_1(dw1), mNormalize(nm), mHandleBorder(hb), mAttMapCount(0), mColorSpace(colorSpace), mWhitening(whitening)
{
	mSrc=src.clone();
	mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);
	mBorderPriorMap = Mat::zeros(src.size(), CV_32FC1);

	if (CL_RGB & colorSpace)
		whitenFeatMap(mSrc, COV_MAT_REG);
	if (CL_Lab & colorSpace)
	{
		Mat lab;
		cvtColor(mSrc, lab, CV_RGB2Lab);
		whitenFeatMap(lab, COV_MAT_REG);
	}
	if (CL_Luv & colorSpace)
	{
		Mat luv;
		cvtColor(mSrc, luv, CV_RGB2Luv);
		whitenFeatMap(luv, COV_MAT_REG);
	}
}

void BMS::computeSaliency(double step)
{
	for (int i=0;i<mFeatureMaps.size();++i)
	{
		Mat bm;
		double max_,min_;
		minMaxLoc(mFeatureMaps[i],&min_,&max_);
		for (double thresh = min_; thresh < max_; thresh += step)
		{
			bm=mFeatureMaps[i]>thresh;
			Mat am = getAttentionMap(bm, mDilationWidth_1, mNormalize, mHandleBorder);
			mSaliencyMap += am;
			mAttMapCount++;
		}
	}
}


cv::Mat BMS::getAttentionMap(const cv::Mat& bm, int dilation_width_1, bool toNormalize, bool handle_border) 
{
	Mat ret=bm.clone();
	int jump;
	if (handle_border)
	{
		for (int i=0;i<bm.rows;i++)
		{
			jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(i,0+jump)!=1)
				floodFill(ret,Point(0+jump,i),Scalar(1),0,Scalar(0),Scalar(0),8);
			jump = BMS_RNG.uniform(0.0,1.0)>0.99 ?BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(i,bm.cols-1-jump)!=1)
				floodFill(ret,Point(bm.cols-1-jump,i),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
		for (int j=0;j<bm.cols;j++)
		{
			jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(0+jump,j)!=1)
				floodFill(ret,Point(j,0+jump),Scalar(1),0,Scalar(0),Scalar(0),8);
			jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(bm.rows-1-jump,j)!=1)
				floodFill(ret,Point(j,bm.rows-1-jump),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
	}
	else
	{
		for (int i=0;i<bm.rows;i++)
		{
			if (ret.at<uchar>(i,0)!=1)
				floodFill(ret,Point(0,i),Scalar(1),0,Scalar(0),Scalar(0),8);
			if (ret.at<uchar>(i,bm.cols-1)!=1)
				floodFill(ret,Point(bm.cols-1,i),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
		for (int j=0;j<bm.cols;j++)
		{
			if (ret.at<uchar>(0,j)!=1)
				floodFill(ret,Point(j,0),Scalar(1),0,Scalar(0),Scalar(0),8);
			if (ret.at<uchar>(bm.rows-1,j)!=1)
				floodFill(ret,Point(j,bm.rows-1),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
	}
	
	ret = ret != 1;
	
	Mat map1, map2;
	map1 = ret & bm;
	map2 = ret & (~bm);

	if (dilation_width_1 > 0)
	{
		dilate(map1, map1, Mat(), Point(-1, -1), dilation_width_1);
		dilate(map2, map2, Mat(), Point(-1, -1), dilation_width_1);
	}
		
	map1.convertTo(map1,CV_32FC1);
	map2.convertTo(map2,CV_32FC1);

	if (toNormalize)
	{
		normalize(map1, map1, 1.0, 0.0, NORM_L2);
		normalize(map2, map2, 1.0, 0.0, NORM_L2);
	}
	else
		normalize(ret,ret,0.0,1.0,NORM_MINMAX);
	return map1+map2;
}

Mat BMS::getSaliencyMap()
{
	Mat ret; 
	normalize(mSaliencyMap, ret, 0.0, 255.0, NORM_MINMAX);
	ret.convertTo(ret,CV_8UC1);
	return ret;
}

void BMS::whitenFeatMap(const cv::Mat& img, float reg)
{
	assert(img.channels() == 3 && img.type() == CV_8UC3);
	
	vector<Mat> featureMaps;
	
	if (!mWhitening)
	{
		split(img, featureMaps);
		for (int i = 0; i < featureMaps.size(); i++)
		{
			normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, NORM_MINMAX);
			medianBlur(featureMaps[i], featureMaps[i], 3);
			mFeatureMaps.push_back(featureMaps[i]);
		}
		return;
	}

	Mat srcF,meanF,covF;
	img.convertTo(srcF, CV_32FC3);
	Mat samples = srcF.reshape(1, img.rows*img.cols);
	calcCovarMatrix(samples, covF, meanF, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_32F);

	covF += Mat::eye(covF.rows, covF.cols, CV_32FC1)*reg;
	SVD svd(covF);
	Mat sqrtW;
	sqrt(svd.w,sqrtW);
	Mat sqrtInvCovF = svd.u * Mat::diag(1.0/sqrtW);

	Mat whitenedSrc = srcF.reshape(1, img.rows*img.cols)*sqrtInvCovF;
	whitenedSrc = whitenedSrc.reshape(3, img.rows);
	
	split(whitenedSrc, featureMaps);

	for (int i = 0; i < featureMaps.size(); i++)
	{
		normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, NORM_MINMAX);
		featureMaps[i].convertTo(featureMaps[i], CV_8U);
		medianBlur(featureMaps[i], featureMaps[i], 3);
		mFeatureMaps.push_back(featureMaps[i]);
	}
}

				
				///////////////// main.cpp content begins //////////////////

void help()
{
	cout<<"Usage: \n"
		<<"BMS <input_path> <output_path> <step_size> <dilation_width1> <dilation_width2> <blurring_std> <color_space> <whitening> <max_dim>\n"
		<<"Press ENTER to continue ..."<<endl;
	getchar();
}


void doWork(
	const string& in_path,
	const string& out_path,
	int sample_step,
	int dilation_width_1,
	int dilation_width_2,
	float blur_std,
	bool use_normalize,
	bool handle_border,
	int colorSpace,
	bool whitening,
	float max_dimension
	)
{
	// Mat image;
 //    image = imread( "../../Code/selena.jpg", 1);

 //    if ( !image.data )
 //    {
 //        printf("No image data \n");
 //        // return -1;
 //    }
 //    namedWindow("Display Image", WINDOW_AUTOSIZE );
 //    imshow("Display Image", image);

 //    waitKey(0);

	if (in_path.compare(out_path)==0)
		cerr<<"output path must be different from input path!"<<endl;
	FileGettor fg(in_path.c_str());
	vector<string> file_list=fg.getFileList();

	clock_t ttt;
	double avg_time=0;
	//#pragma omp parallel for
	for (int i=0;i<file_list.size();i++)
	{
		/* get file name */
		string ext=getExtension(file_list[i]);
		if (!(ext.compare("jpg")==0 || ext.compare("jpeg")==0 || ext.compare("JPG")==0 || ext.compare("tif")==0 || ext.compare("png")==0 || ext.compare("bmp")==0))
			continue;
		//cout<<file_list[i]<<"...";

		/* Preprocessing */
		Mat src=imread(in_path+file_list[i]);
		Mat src_small;
		float w = (float)src.cols, h = (float)src.rows;
		float maxD = max(w,h);
		if (max_dimension < 0)
			resize(src,src_small,Size((int)(MAX_IMG_DIM*w/maxD),(int)(MAX_IMG_DIM*h/maxD)),0.0,0.0,INTER_AREA);// standard: width: 600 pixel
		else
			resize(src, src_small, Size((int)(max_dimension*w / maxD), (int)(max_dimension*h / maxD)), 0.0, 0.0, INTER_AREA);


		/* Computing saliency */
		ttt=clock();

		BMS bms(src_small, dilation_width_1, use_normalize, handle_border, colorSpace, whitening);
		bms.computeSaliency((double)sample_step);
		
		Mat result=bms.getSaliencyMap();

		/* Post-processing */

		if (dilation_width_2 > 0)
			dilate(result, result, Mat(), Point(-1, -1), dilation_width_2);
		if (blur_std > 0)
		{
			int blur_width = (int)MIN(floor(blur_std) * 4 + 1, 51);
			GaussianBlur(result, result, Size(blur_width, blur_width), blur_std, blur_std);
		}

		
		
		ttt=clock()-ttt;
		float process_time=(float)ttt/CLOCKS_PER_SEC;
		avg_time+=process_time;
		//cout<<"average_time: "<<avg_time/(i+1)<<endl;

		/* Save the saliency map*/
		resize(result,result,src.size());
		imwrite(out_path+rmExtension(file_list[i])+".png",result);		
	}
	cout << "average_time: " << avg_time / file_list.size() << endl;
}


int main(int args, char** argv)
{
	if (args < 9)
	{
		cout<<"wrong number of input arguments."<<endl;
		help();
		return 1;
	}

	/* initialize system parameters */
	string INPUT_PATH		=	argv[1];
	string OUTPUT_PATH		=	argv[2];
	int SAMPLE_STEP			=	atoi(argv[3]);//8: delta

	/*Note: we transform the kernel width to the equivalent iteration 
	number for OpenCV's **dilate** and **erode** functions**/	
	int DILATION_WIDTH_1	=	(atoi(argv[4])-1)/2;//3: omega_d1
	int DILATION_WIDTH_2	=	(atoi(argv[5])-1)/2;//11: omega_d2

	float BLUR_STD			=	(float)atof(argv[6]);//20: sigma	
	bool NORMALIZE			=	1 /*atoi(argv[7])*/;//1: whether to use L2-normalization
	bool HANDLE_BORDER		=	0 /*atoi(argv[8])*/;//0: to handle the images with artificial frames
	int COLORSPACE			=	atoi(argv[7]);//
	bool WHITENING			=	atoi(argv[8]);
	
	float MAX_DIM			=	-1.0f;
	if (args > 9)
		MAX_DIM				=	(float)atof(argv[9]);

	doWork(INPUT_PATH,OUTPUT_PATH,SAMPLE_STEP,DILATION_WIDTH_1,DILATION_WIDTH_2,BLUR_STD,NORMALIZE,HANDLE_BORDER, COLORSPACE, WHITENING, MAX_DIM);

	// Mat image;
 //    image = imread( "../../Code/selena.jpg", 1);

 //    if ( !image.data )
 //    {
 //        printf("No image data \n");
 //        // return -1;
 //    }
 //    namedWindow("Display Image", WINDOW_AUTOSIZE );
 //    imshow("Display Image", image);

 //    waitKey(0);

	return 0;
}