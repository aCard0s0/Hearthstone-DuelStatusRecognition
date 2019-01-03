
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




//cv::Mat IMAGES[] = {
//	cv::imread("resources/1920x1080/druid_v.png"),
//	cv::imread("resources/1920x1080/hunter_v.png"),
//	cv::imread("resources/1920x1080/mage_v.png"),
//	cv::imread("resources/1920x1080/paladin_v.png"),
//	cv::imread("resources/1920x1080/priest_v.png"),
//	cv::imread("resources/1920x1080/rogue_v.png"),
//	cv::imread("resources/1920x1080/warlock_v.png")
//};


/*
	The list of images to be analyzed
*/
cv::Mat IMAGES[] = {
	
	cv::imread("resources/1920x1080/druid_v.png"),
	cv::imread("resources/1920x1080/hunter_v.png"),
	cv::imread("resources/1920x1080/mage_v.png"),
	cv::imread("resources/1920x1080/paladin_v.png"),
	cv::imread("resources/1920x1080/priest_v.png"),
	cv::imread("resources/1920x1080/rogue_v.png"),
	cv::imread("resources/1920x1080/warlock_v.png"),

	cv::imread("resources/1280x720/resized_druid_v.png"),
	cv::imread("resources/1280x720/resized_hunter_v.png"),
	cv::imread("resources/1280x720/resized_mage_v.png"),
	cv::imread("resources/1280x720/druid_v.jpg"),
	cv::imread("resources/1280x720/hunter_v.jpg"),
	cv::imread("resources/1280x720/mage_v.jpg"),
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
	/*cv::imread("resources/icons/druid_icon.png"),
	cv::imread("resources/icons/hunter_icon.png"),
	cv::imread("resources/icons/mage_icon.png"),
	cv::imread("resources/icons/paladin_icon.png"),
	cv::imread("resources/icons/priest_icon.png"),
	cv::imread("resources/icons/rogue_icon.png"),
	cv::imread("resources/icons/warlock_icon.png"),
	cv::imread("resources/icons/warrior_icon.png"),
	cv::imread("resources/icons/shaman_icon.png")*/
	cv::imread("resources/icons/Druid_canny.jpg"),
	cv::imread("resources/icons/Hunter_canny.jpg"),
	cv::imread("resources/icons/Mage_canny.jpg"),
	cv::imread("resources/icons/Paladin_canny.jpg"),
	cv::imread("resources/icons/Priest_canny.jpg"),
	cv::imread("resources/icons/Rogue_canny.jpg"),
	cv::imread("resources/icons/Warlock_canny.jpg"),
	cv::imread("resources/icons/Warrior_canny.jpg"),
	cv::imread("resources/icons/Shaman_canny.jpg")
};
int const NUM_CLASSES = (int)(sizeof(CLASS_TEMPLATES) / sizeof(CLASS_TEMPLATES[0]));

/*
	Templates for "Victory" and "Defeat"
*/
cv::Mat RESULT_TEMPLATES[] = {
	cv::imread("resources/icons/victory_canny.png"),
	cv::imread("resources/icons/defeat_canny.png")
};
int const NUM_RESULTS = (int)(sizeof(RESULT_TEMPLATES) / sizeof(RESULT_TEMPLATES[0]));

/* 
	Methods used on the template matching
*/
int MATCH_METHODS[] = { cv::TM_CCOEFF_NORMED /*, cv::TM_CCORR_NORMED , cv::TM_CCOEFF,*/ };
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


cv::Mat apply_canny(cv::Mat original, int low_threshold = 86, int thresh_ratio = 3)
{
	// Convert the image to grayscale
	cv::Mat image_gray;
	cvtColor(original, image_gray, cv::COLOR_RGB2GRAY);

	// Reduce noise with a kernel 3x3
	cv::Mat detected_edges;
	cv::blur(image_gray, detected_edges, cv::Size(3, 3));

	// Canny detector
	//int low_threshold = global_low_threshold; // DEBUG -------------------------
	int kernel_size = 3;
	cv::Canny(detected_edges, detected_edges, low_threshold, low_threshold * thresh_ratio, kernel_size);

	// Using Canny's output as a mask, we display our result
	cv::Mat img_canny;
	img_canny = cv::Scalar::all(0);
	original.copyTo(img_canny, detected_edges);

	return img_canny;
}


/*

	type:
		0 = class icon
		1 = result
*/
cv::Point get_image_center(cv::Mat image, int type)
{
	
	// Templates used are rectangles
	// Part of the class icon is cropped because it has curves
	int y_offset;
	// The vertical center of the class image in comparison with the full image height
	double placement_ratio;

	if (type == 0)
	{
		y_offset = 20;
		placement_ratio = 0.375;
	}
	else
	{
		y_offset = 0;
		placement_ratio = 0.5805;
	}
	
	cv::Point img_center;
	img_center.x = (int)(image.cols / 2);
	img_center.y = (int)(image.rows * placement_ratio) + y_offset;

	return img_center;
}


/*
	Best matched template
*/
void get_best_match(double worst_case, int *tmpl_matched, double templ_score[], int size)
{
	double best_val = worst_case;
	for (int tmpl_idx = 0; tmpl_idx < size; tmpl_idx++)
	{
		if (templ_score[tmpl_idx] < best_val)
		{
			best_val = templ_score[tmpl_idx];
			*tmpl_matched = tmpl_idx;
		}
	}
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

		// Validate and correct dimensions
		if (original_img.rows != 720)
		{
			int new_width = (original_img.cols * 720) / original_img.rows;
			cv::resize(original_img, original_img, cv::Size(new_width, 720));
		}

		cv::imshow(window_name, original_img);

		// The center of the class icon
		cv::Point img_center = get_image_center(original_img, 0);

		// Apply canny filter
		cv::Mat img_canny = apply_canny(original_img);
		
		// Use template matching with all class templates
		for (int tmpl_idx = 0; tmpl_idx < NUM_CLASSES; tmpl_idx++)
		{
			// Create the result matrix
			cv::Mat result;
			cv::Mat tmpl = CLASS_TEMPLATES[tmpl_idx];

			int result_cols = original_img.cols - tmpl.cols + 1;
			int result_rows = original_img.rows - tmpl.rows + 1;
			result.create(result_rows, result_cols, CV_32FC1);

			// Create canny template
			//cv::Mat canny_template;
			//canny_template = to_canny(CLASS_TEMPLATES[tmpl_idx]);
			//canny_template = CLASS_TEMPLATES[tmpl_idx];
			
			// Write to file
			//string fn = "resources/icons/" + get_class_name(tmpl_idx) + "_canny.jpg";
			//cv::imwrite(fn, canny_template);

			// Do the matching
			//cv::matchTemplate(img_canny, canny_template, result, MATCH_METHODS[0]);
			cv::matchTemplate(img_canny, tmpl, result, MATCH_METHODS[0]);

			// Localizing the best match with minMaxLoc
			double min_val, match_val;
			cv::Point min_loc, max_loc, match_loc;

			minMaxLoc(result, &min_val, &match_val, &min_loc, &max_loc, cv::Mat());
			match_loc = max_loc;
			
			// Template center
			cv::Point tmpl_center;
			tmpl_center.x = (int) (tmpl.cols / 2 + match_loc.x);
			tmpl_center.y = (int) (tmpl.rows / 2 + match_loc.y);

			// Distance from template center to class icon center
			double distance = sqrt( pow(abs(img_center.x - tmpl_center.x),2) + pow(abs(img_center.y - tmpl_center.y),2) );

			// Info output
			cout << "Template[" << tmpl_idx << "]: " << get_class_name(tmpl_idx) << endl;
			cout << "max_val: " << match_val << endl;
			cout << "Distance to center: " << distance << endl;
			cout << "score: " << distance - match_val * 10 << endl;
			cout << "----------------" << endl;

			// Visual output
			/*
			cv::Mat display_img = original_img.clone();
			cv::rectangle(display_img, match_loc, cv::Point(match_loc.x + tmpl.cols, match_loc.y + tmpl.rows), cv::Scalar::all(0), 2, 8, 0);
			cv::imshow(window_name, display_img);
			//cv::imshow("canny img", img_canny);
			cv::imshow("templ", CLASS_TEMPLATES[tmpl_idx]); 
			cv::waitKey(0);
			*/
			// Save result
			templ_score[tmpl_idx] = distance - match_val * 10;
		}

		int tmpl_matched;
		get_best_match((double)original_img.cols, &tmpl_matched, templ_score, NUM_CLASSES);

		cout << "Prediction: " << get_class_name(tmpl_matched) << "(" << tmpl_matched << ")" << endl;
		cout << "\n#######################\n" << endl;
		// Pause before next image
		cv::waitKey(0);
	}
}



/*
	Uses template matching to find the match result
*/
void template_match_result(int, void*)
{
	double templ_score[NUM_RESULTS];

	// Analyze each image
	for (int idx = 0; idx < NUM_IMAGES; idx++)
	{
		cv::Mat original_img = IMAGES[idx].clone();

		// Validate and correct dimensions
		if (original_img.rows != 720)
		{
			int new_width = (original_img.cols * 720) / original_img.rows;
			cv::resize(original_img, original_img, cv::Size(new_width, 720));
		}

		cv::imshow(window_name, original_img);

		// The center of the result icon
		cv::Point img_center = get_image_center(original_img, 1);

		// Apply canny filter
		cv::Mat img_canny = apply_canny(original_img);

		// Use template matching with victory/defeat templates
		for (int tmpl_idx = 0; tmpl_idx < NUM_RESULTS; tmpl_idx++)
		{
			// Create the result matrix
			cv::Mat result;
			cv::Mat tmpl = RESULT_TEMPLATES[tmpl_idx];

			int result_cols = original_img.cols - tmpl.cols + 1;
			int result_rows = original_img.rows - tmpl.rows + 1;
			result.create(result_rows, result_cols, CV_32FC1);

			//Create canny template
			//cv::Mat canny_template;
			//canny_template = apply_canny(RESULT_TEMPLATES[tmpl_idx]);
			//canny_template = RESULT_TEMPLATES[tmpl_idx];

			// Write to file
			//string vd = "defeat";
			//if (tmpl_idx == 0)
			//	vd = "victory";

			//string fn = "resources/icons/" + vd + "_canny.png";
			//cv::imwrite(fn, canny_template);

			// Do the matching
			//cv::matchTemplate(img_canny, canny_template, result, MATCH_METHODS[0]);
			cv::matchTemplate(img_canny, tmpl, result, MATCH_METHODS[0]);

			// Localizing the best match with minMaxLoc
			double min_val, match_val;
			cv::Point min_loc, max_loc, match_loc;

			minMaxLoc(result, &min_val, &match_val, &min_loc, &max_loc, cv::Mat());
			match_loc = max_loc;

			// Template center
			cv::Point tmpl_center;
			tmpl_center.x = (int)(tmpl.cols / 2 + match_loc.x);
			tmpl_center.y = (int)(tmpl.rows / 2 + match_loc.y);

			// Distance from template center to class icon center
			double distance = sqrt(pow(abs(img_center.x - tmpl_center.x), 2) + pow(abs(img_center.y - tmpl_center.y), 2));

			// Info output
			cout << "Template: " << tmpl_idx << " --- (0=vic,1=def)" << endl;
			cout << "max_val: " << match_val << endl;
			cout << "Distance to center: " << distance << endl;
			cout << "score: " << distance - match_val * 10 << endl;
			cout << "----------------" << endl;

			// Visual output
			
			cv::Mat display_img = original_img.clone();
			cv::rectangle(display_img, match_loc, cv::Point(match_loc.x + tmpl.cols, match_loc.y + tmpl.rows), cv::Scalar::all(0), 2, 8, 0);
			cv::imshow(window_name, display_img);
			//cv::imshow("canny img", img_canny);
			cv::imshow("templ", RESULT_TEMPLATES[tmpl_idx]);
			
			// Save result
			templ_score[tmpl_idx] = distance - match_val * 10;
			cv::waitKey(0);
		}

		int tmpl_matched;
		get_best_match((double)original_img.cols, &tmpl_matched, templ_score, NUM_RESULTS);

		string prediction = "Victory";
		if (tmpl_matched == 1)
			prediction = "Defeat";
		cout << "Prediction: " << prediction << " (" << tmpl_matched << ")" << endl;
		cout << "\n#######################\n" << endl;
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
	//template_match_result(0, 0);
	template_match_class(0, 0);

	// Wait for user input to exit
	cv::waitKey(0);

	return 0;
}