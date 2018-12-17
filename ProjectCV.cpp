// ProjectCV.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//	Link de interesse:
//		https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
//		https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
//		https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

#include "pch.h"
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

int main()
{
	cv::Mat image;

	image = cv::imread("resources/druid_d.jpg", cv::IMREAD_UNCHANGED);

	cv::namedWindow("Imagem Original", cv::WINDOW_AUTOSIZE);
	cv::imshow("Imagem Original", image);


	cv::waitKey(0);					// Waiting
	cv::destroyAllWindows();		// Destroy the windows
}