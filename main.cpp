
#include "pch.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

//Global Variables
char* image_window = (char*)"Source Image";
char* result_window = (char*)"Result window";
int global_match_method;

int edgeThresh = 1;
int global_low_threshold;
int const max_lowThreshold = 100;
char* window_name = (char*)"Edge Map";



/*
cv::Mat images[] = {	
	cv::imread("resources/1920x1080/druid_v.png"),
	cv::imread("resources/1920x1080/hunter_v.png"),
	cv::imread("resources/1920x1080/mage_v.png"),
	cv::imread("resources/1920x1080/paladin_v.png"),
	cv::imread("resources/1920x1080/priest_v.png"),
	cv::imread("resources/1920x1080/rogue_v.png"),
	cv::imread("resources/1920x1080/warlock_v.png")
};
*/

/*
	The list of images to be analyzed
*/
cv::Mat IMAGES[] = {
	/*cv::imread("resources/1280x720/resized_druid_v.png"),
	cv::imread("resources/1280x720/resized_hunter_v.png"),
	cv::imread("resources/1280x720/resized_mage_v.png"),*/
	/*cv::imread("resources/1280x720/druid_v.jpg"),
	cv::imread("resources/1280x720/hunter_v.jpg"),
	cv::imread("resources/1280x720/mage_v.jpg"),*/
	cv::imread("resources/1280x720/resized_druid_d.png"),
	cv::imread("resources/1280x720/resized_hunter_d.png"),
	cv::imread("resources/1280x720/resized_mage_d.png"),
	cv::imread("resources/1280x720/paladin_v.jpg"),
	cv::imread("resources/1280x720/priest_v.jpg"),
	cv::imread("resources/1280x720/rogue_v.jpg"),
	cv::imread("resources/1280x720/warlock_v.jpg")
};
int const NUM_IMAGES = (int)(sizeof(IMAGES) / sizeof(IMAGES[0]));

/*
	Template of the classes
*/
cv::Mat CLASS_TEMPLATES[] = {
	cv::imread("resources/icons/druid_icon.png"),
	cv::imread("resources/icons/hunter_icon.png"),
	cv::imread("resources/icons/mage_icon.png"),
	cv::imread("resources/icons/paladin_icon.png"),
	cv::imread("resources/icons/priest_icon.png"),
	cv::imread("resources/icons/rogue_icon.png"),
	cv::imread("resources/icons/warlock_icon.png"),
	cv::imread("resources/icons/warrior_icon.png"),
	cv::imread("resources/icons/shaman_icon.png")
};
int const NUM_CLASSES = (int)(sizeof(CLASS_TEMPLATES) / sizeof(CLASS_TEMPLATES[0]));

/*
	Templates for "Victory" and "Defeat"
*/
cv::Mat RESULT_TEMPLATES[] = {
	cv::imread("resources/icons/victory_icon.jpg"),
	cv::imread("resources/icons/defeat_icon.png")
};


/* 
	Methods used on the template matching
*/
int MATCH_METHODS[] = { /*cv::TM_CCORR_NORMED , cv::TM_CCOEFF,*/ cv::TM_CCOEFF_NORMED };
int const NUM_METHODS = (int)(sizeof(MATCH_METHODS) / sizeof(MATCH_METHODS[0]));

/* 
	Translates the int index to the name of the matching method 
*/
string match_method_name(int method)
{
	switch (method)
	{
		case 0:	return "SQDIFF";
		case 1: return "SQDIFF NORMED";
		case 2:	return "TM CCORR";
		case 3: return "TM CCORR NORMED";
		case 4: return "TM COEFF";
		case 5: return "TM COEFF NORMED";
		default: return "INVALID NUM";
	}
}

/*
	Translates the int index to the name of the class 	
*/
string get_class_name (int i)
{
	switch (i)
	{
		case 0:	return "Druid";
		case 1: return "Hunter";
		case 2:	return "Mage";
		case 3: return "Paladin";
		case 4: return "Priest";
		case 5: return "Rogue";
		case 6: return "Warlock";
		case 7: return "Warrior";
		case 8: return "Shaman";
		default: return "INVALID NUM";
	}
}


cv::Mat to_canny(cv::Mat original)
{
	// Convert the image to grayscale
	cv::Mat image_gray;
	cvtColor(original, image_gray, cv::COLOR_RGB2GRAY);

	// Reduce noise with a kernel 3x3
	//cv::Mat detected_edges;
	//cv::blur(image_gray, detected_edges, cv::Size(3, 3));
	cv::Mat detected_edges = image_gray; // DEBUG -------------------

	// Canny detector
	int low_threshold = 86;
	//int low_threshold = global_low_threshold; // DEBUG -------------------------
	int thrsh_ratio = 3;
	int kernel_size = 3;
	cv::Canny(detected_edges, detected_edges, low_threshold, low_threshold * thrsh_ratio, kernel_size);

	// Using Canny's output as a mask, we display our result
	cv::Mat img_canny;
	img_canny = cv::Scalar::all(0);
	original.copyTo(img_canny, detected_edges);

	return img_canny;
}


/*
	Uses template matching to find which class was played
*/
void template_match_class(int, void*)
{	
	double templ_score[NUM_CLASSES];

	for (int idx = 0; idx < NUM_IMAGES; idx++)
	{

		cv::Mat original_img = IMAGES[idx].clone();
		cv::Mat img_canny = to_canny(IMAGES[idx]);

		//for (int tmpl_idx = idx; tmpl_idx == idx; tmpl_idx++) ----------------DEBUG
		for (int tmpl_idx = 0; tmpl_idx < NUM_CLASSES; tmpl_idx++)
		{
			// Create the result matrix
			cv::Mat result;
			int result_cols = original_img.cols - CLASS_TEMPLATES[tmpl_idx].cols + 1;
			int result_rows = original_img.rows - CLASS_TEMPLATES[tmpl_idx].rows + 1;
			result.create(result_rows, result_cols, CV_32FC1);

			// Create canny template
			cv::Mat canny_template;
			canny_template = to_canny(CLASS_TEMPLATES[tmpl_idx]);
			
			// Do the matching
			int match_method = MATCH_METHODS[0];
			cv::matchTemplate(img_canny, canny_template, result, match_method);

			// Localizing the best match with minMaxLoc
			double min_val; double max_val;
			cv::Point min_loc; cv::Point max_loc;
			cv::Point match_loc;
			minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc, cv::Mat());

			// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
			double result_val;
			if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED)
			{
				match_loc = min_loc;
				result_val = min_val;
			}
			else
			{
				match_loc = max_loc;
				result_val = max_val;
			}

			// We dont use the icon from the top 
			// Because templates used are rectangles and the class icons have curves
			// So an offset is used
			int y_offset = 10;

			// The center of the class icon
			cv::Point img_center;
			img_center.x = (int) (img_canny.cols / 2);
			img_center.y = (int) (img_canny.rows * 0.375) + y_offset;  

			// Template center
			cv::Point templ_center;
			templ_center.x = (int) (canny_template.cols / 2 + match_loc.x);
			templ_center.y = (int) (canny_template.rows / 2 + match_loc.y);

			// Distance from template center to class icon center
			double distance = sqrt( abs(img_center.x - templ_center.x)^2 + abs(img_center.y - templ_center.y)^2 );

			// Info output
			cout << "Image[" << idx << "]: " << get_class_name(idx) << endl;
			cout << "Template[" << tmpl_idx << "]: " << get_class_name(tmpl_idx) << endl;
			//cout << "Method[" << match_method << "]: " << match_method_name(match_method) << endl;
			//cout << "Thresh: " << global_low_threshold << endl;
			cout << "max_val: " << max_val << endl;
			cout << "Distance to center: " << distance << endl;
			cout << "----------------" << endl;

			// Visual output
			/*cv::Mat display_img = original_img.clone();
			cv::rectangle(display_img, matchLoc, cv::Point(matchLoc.x + canny_template.cols, matchLoc.y + canny_template.rows), cv::Scalar::all(0), 2, 8, 0);
			cv::imshow(window_name, display_img);
			cv::imshow("canny img", img_canny);
			cv::imshow("canny templ", canny_template); */
			// Save result
			templ_score[tmpl_idx] = distance;

			// Pause before next template
			//cv::waitKey(0);	
		}

		double best_val = original_img.cols;
		int tmpl_matched;
		for (int tmpl_idx = 0; tmpl_idx < NUM_CLASSES; tmpl_idx++)
		{
			double score = templ_score[tmpl_idx];
			if (score < best_val)
			{
				best_val = score;
				tmpl_matched = tmpl_idx;
			}
		}
		
		cout << "Prediction: " << tmpl_matched << endl;
		cout << "#######################" << endl;
		// Pause before next image
		cv::waitKey(0);
	}
}


/*
	Uses template matching to find the match result
*/
void template_match_result(int, void*)
{
	// Victory / Defeat
	double templ_score[2];

	for (int idx = 0; idx < NUM_IMAGES; idx++)
	{
		cv::Mat original_img = IMAGES[idx].clone();
		cv::Mat img_canny = to_canny(IMAGES[idx]);

		for (int tmpl_idx = 0; tmpl_idx < 2; tmpl_idx++)
		{
			// Create the result matrix
			cv::Mat result;
			int result_cols = original_img.cols - RESULT_TEMPLATES[tmpl_idx].cols + 1;
			int result_rows = original_img.rows - RESULT_TEMPLATES[tmpl_idx].rows + 1;
			result.create(result_rows, result_cols, CV_32FC1);

			// Create canny template
			cv::Mat canny_template;
			canny_template = to_canny(RESULT_TEMPLATES[tmpl_idx]);

			// Do the matching
			int match_method = MATCH_METHODS[0];
			cv::matchTemplate(img_canny, canny_template, result, match_method);

			// Localizing the best match with minMaxLoc
			double min_val; double max_val;
			cv::Point min_loc; cv::Point max_loc;
			cv::Point match_loc;
			minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc, cv::Mat());

			// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
			double result_val;
			if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED)
			{
				match_loc = min_loc;
				result_val = min_val;
			}
			else
			{
				match_loc = max_loc;
				result_val = max_val;
			}

			// The center of the result icon
			cv::Point img_center;
			img_center.x = (int)(img_canny.cols / 2);
			img_center.y = (int)(img_canny.rows * 0.5805);

			// Template center
			cv::Point templ_center;
			templ_center.x = (int)(canny_template.cols / 2 + match_loc.x);
			templ_center.y = (int)(canny_template.rows / 2 + match_loc.y);

			// Distance from template center to class icon center
			double distance = sqrt(abs(img_center.x - templ_center.x) ^ 2 + abs(img_center.y - templ_center.y) ^ 2);

			// Info output
			cout << "Image[" << idx << "]: " << get_class_name(idx) << endl;
			cout << "victory(0) / defeat(1) : " << tmpl_idx << endl;
			//cout << "Method[" << match_method << "]: " << match_method_name(match_method) << endl;
			cout << "max_val: " << max_val << endl;
			cout << "Thresh: " << global_low_threshold << endl;
			cout << "Distance to center: " << distance << endl;
			cout << "----------------" << endl;

			// Visual output
			cv::Mat display_img = original_img.clone();
			cv::rectangle(display_img, match_loc, cv::Point(match_loc.x + canny_template.cols, match_loc.y + canny_template.rows), cv::Scalar::all(0), 2, 8, 0);
			cv::imshow(window_name, display_img);
			cv::imshow("canny img", img_canny);
			cv::imshow("canny templ", canny_template);

			// Save result
			templ_score[tmpl_idx] = distance;

			// Pause before next template
			//cv::waitKey(0);	
		}

		double best_val = original_img.cols;
		int tmpl_matched;
		for (int tmpl_idx = 0; tmpl_idx < 2; tmpl_idx++)
		{
			double score = templ_score[tmpl_idx];
			if (score < best_val)
			{
				best_val = score;
				tmpl_matched = tmpl_idx;
			}
		}
		string res;
		if (tmpl_matched == 0)
			res = "Victory";
		else
			res = "Defeat";

		cout << "Prediction: " << res << endl;
		cout << "#######################" << endl;
		// Pause before next image
		cv::waitKey(0);
	}
}



int main(int argc, char** argv)
{
	// Create a window
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

	// Canny threshold trackbar
	//cv::createTrackbar("Min Threshold:", window_name, &global_low_threshold, max_lowThreshold, template_match_class);
	cv::createTrackbar("Min Threshold:", window_name, &global_low_threshold, max_lowThreshold, template_match_result);

	// Template matching method trackbar
	int max_trackbar = 5;
	char* trackbar_label = (char*)"Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	cv::createTrackbar(trackbar_label, window_name, &global_match_method, max_trackbar, template_match_result);

	// Matching
	template_match_result(0, 0);
	//template_match_class(0, 0);

	// Wait for user input to exit
	cv::waitKey(0);

	return 0;
}