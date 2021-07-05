#pragma once

class banana_dataset {
#pragma once
	/*Created by Alberto Chimenti on 20th May 2021*/

public:
	banana_dataset(std::string filepath);

	std::string fname(int index);
	std::vector<int> bbox(int index);
	int size();

	void read_csv(std::string filepath);

protected:
	std::vector<std::string> filename;
	std::vector<int> label, xmin, ymin, xmax, ymax;
};


class NetInput {

public:
	NetInput(std::string filename);
	cv::Mat GetImage();
	std::vector<std::vector<int>> getgrid(int ws, int stride);
	cv::Mat GetPatch(std::vector<int> bbox);
	//cv::Mat GetPatch(int index, int window_size);

protected:
	cv::Mat in_img, patch;
	std::vector<cv::Point> coordinates;

};

int findlabel(std::vector<int> orig, std::vector<int> curr, double& overlap, double thr_overlap = 0.3);

std::vector<int> center_from_bbox(std::vector<int> bbox);