/*Created by Alberto Chimenti on 20th May 2021*/


// include openCV and standard headers
#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "banana_utils.h"



int main(int argc, char** argv) {
	/*
	Inputs: 

	data_folder	---------------> (std::string) Path of the data folder containing: "images" folder and "label.csv" file

	(optional)window_size -----> (int) Set window_size (default = 50)

	(optional)stride ----------> (int) Set stride of the sliding window (default = 25)

	(optional)overlap ---------> (double) Set minimum percentage overlap to consider the image portion label equal to 1 (default = 0.3)

	(optional)train_mode ------> (std::string) Set equal to keyword "train" to allow sampling of additional patches for better label distribution

	(optional)additional ------> (int) If train_mode is on then one can set a number of requested additional samples taken in the neighborhood of the object box
	*/


	// defining input variables
	static std::string data_folder;
	static int window_size, stride, additional;
	static double overlap;
	static bool train_mode;

	if (argv[1]) data_folder = argv[1];
	else { std::cout << "***Please provide a data folder path!...exiting" << std::endl; exit(EXIT_FAILURE); };

	if (argc>1 && argv[2]) window_size = std::stoi(argv[2]);
	else window_size = 50;

	if (argc>2 && argv[3]) stride = std::stoi(argv[3]);
	else stride = 25;

	if (argc>3 && argv[4]) overlap = std::stod(argv[4]);
	else overlap = 0.2;

	if (argc > 4 && argv[5] == std::string("train")) train_mode = true;
	else train_mode = false;

	if (argc > 5 && argv[6]) additional = std::stoi(argv[6]);
	else additional = 0;

	// check inputs
	if (!std::filesystem::exists(std::filesystem::path(data_folder))) { std::cout << "***Cannot find the data folder...exiting" << std::endl; exit(EXIT_FAILURE); };
	if ((overlap <= 0.0) || (overlap > 1.0)) { std::cout << "***Selected overlap is not between 0 and 1...exiting" << std::endl; exit(EXIT_FAILURE); };
	if ((window_size <= 1) || (window_size > 256)) { std::cout << "***Selected window_size is not valid...exiting" << std::endl; exit(EXIT_FAILURE); };
	if ((stride <= 0) || (stride > 128)) { std::cout << "***Selected stride is not valid...exiting" << std::endl; exit(EXIT_FAILURE); };
	if ((additional < 0)) { std::cout << "***Selected additional number of samples is not valid...exiting" << std::endl; exit(EXIT_FAILURE); };

	std::cout << "Running code with the following parameters..." << std::endl;
	std::cout << "\tPreprocessing folder: " << data_folder << std::endl;
	std::cout << "\tWindow size: " << window_size << std::endl;
	std::cout << "\tStride: " << stride << std::endl;
	std::cout << "\tOverlap: " << overlap << std::endl;
	if (train_mode && additional!=0) { std::cout << "\tAdditional samples: " << additional << std::endl; }


	// load dataset header
    banana_dataset data(data_folder+"/label.csv");
	// open new header as output file
	std::ofstream outfile (data_folder+"/label_processed_" + std::to_string(window_size) + ".csv", std::ofstream::trunc);
	// remove old folder if present and create new one
	std::filesystem::remove_all(data_folder + "/images_processed_" + std::to_string(window_size));
	std::filesystem::create_directory(data_folder + "/images_processed_" + std::to_string(window_size));

	// write column names
	outfile << "img_name,label,reg_label,xmin,ymin,xmax,ymax" << std::endl;
	std::string fname; //placeholder init
	double actual_overlap = 0.; //placeholder float window overlap with bbox

	for (int idx = 0; idx < data.size(); idx++) {

		NetInput src(data_folder + "/images/" + data.fname(idx));
		std::vector<std::vector<int>> grid = src.getgrid(window_size, stride);

		// display process bar
		std::cout << idx << "...";

		if (train_mode) {
			// write banana_centered patch for training examples
			//...if possible...
			std::vector<int> center = center_from_bbox(data.bbox(idx));
			std::vector<int> box = { center[0] - window_size/2, center[1] - window_size/2, center[0] + window_size/2, center[1] + window_size/2 };
			if (box[0]<0 || box[1]<0 || box[2]>src.GetImage().size[0] || box[3]>src.GetImage().size[0]) { continue; }
			else {	
				outfile << data.fname(idx) << "," << findlabel(data.bbox(idx), box, actual_overlap, overlap) << "," << actual_overlap << "," << box[0] << "," << box[1] << "," << box[2] << "," << box[3] << std::endl;
				cv::imwrite(data_folder + "/images_processed_" + std::to_string(window_size) + "/" + data.fname(idx), src.GetPatch(box));
			}
		}

		for (int i = 0; i < grid.size(); i++) {
			// new filename
			fname = std::to_string(idx) + "_" + std::to_string(i) + ".png";
			// write to file
			outfile << fname << "," << findlabel(data.bbox(idx), grid[i], actual_overlap, overlap) << "," << actual_overlap << "," << grid[i][0] << "," << grid[i][1] << "," << grid[i][2] << "," << grid[i][3] << std::endl;
			// write image patch
			cv::imwrite(data_folder + "/images_processed_" + std::to_string(window_size) + "/" + fname, src.GetPatch(grid[i]));
		}

		// add more positive label training samples by random sampling close to the roi
		if (train_mode && additional != 0) {
			int rand_x, rand_y;
			// set seed for consistent results
			std::srand(std::time(0));
			// get central roi's central point
			std::vector<int> center = center_from_bbox(data.bbox(idx));
			// set min/max values of the increment interval
			int extreme = static_cast<int>((data.bbox(idx)[2] - data.bbox(idx)[0]) / 2);
			for (int r = 0; r < additional; r++) {
				rand_x = -extreme + (std::rand() % (2 * extreme + 1));
				rand_y = -extreme + (std::rand() % (2 * extreme + 1));
				std::vector<int> n_box = { center[0] + rand_x - window_size / 2, center[1] + rand_y - window_size / 2, center[0] + rand_x + window_size / 2, center[1] + rand_y + window_size / 2 };

				if (n_box[0]<0 || n_box[1]<0 || n_box[2]>src.GetImage().size[0] || n_box[3]>src.GetImage().size[0]) { continue; }
				else {
					fname = std::to_string(idx) + "_" + std::to_string(r + grid.size()) + ".png";
					outfile << fname << "," << findlabel(data.bbox(idx), n_box, actual_overlap, overlap) << "," << actual_overlap << "," << n_box[0] << "," << n_box[1] << "," << n_box[2] << "," << n_box[3] << std::endl;
					cv::imwrite(data_folder + "/images_processed_" + std::to_string(window_size) + "/" + fname, src.GetPatch(n_box));
				}
			}
		}

	}
	std::cout << "Done!" << std::endl;
	outfile.close();
	//cv::destroyAllWindows();
	return 0;
};