/*Created by Alberto Chimenti on 20th May 2021*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept> // std::runtime_error
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "banana_utils.h"


/*###########################################################################*/
/*                           DATASET CLASS                                   */


banana_dataset::banana_dataset(std::string filepath) { read_csv(filepath); };

std::string banana_dataset::fname(int index) { return filename[index]; };

std::vector<int> banana_dataset::bbox(int index) {
    std::vector<int> result = { xmin[index], ymin[index], xmax[index], ymax[index] };
    return result;
};

int banana_dataset::size() { 
    return filename.size(); 
};

void banana_dataset::read_csv(std::string filepath) {

    // Create an input filestream
    std::ifstream myFile(filepath);
    //define placeholders for reading from file
    std::string line, temp;

    // Make sure the file is open
    if (!myFile.is_open()) throw std::runtime_error("Could not open file");

    std::getline(myFile, line); //skip first header line

    // Read data, line by line
    while (std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        std::getline(ss, temp, ',');
        filename.push_back(temp);
        std::getline(ss, temp, ',');
        label.push_back(std::stoi(temp));
        std::getline(ss, temp, ',');
        xmin.push_back(std::stoi(temp));
        std::getline(ss, temp, ',');
        ymin.push_back(std::stoi(temp));
        std::getline(ss, temp, ',');
        xmax.push_back(std::stoi(temp));
        std::getline(ss, temp, ',');
        ymax.push_back(std::stoi(temp));
    }
    // Close file
    myFile.close();

};


/*###########################################################################*/
/*                           IMAGE CLASS                                     */

NetInput::NetInput(std::string filename) {

    in_img = cv::imread(cv::samples::findFile(filename));
    //if fail to read the image
    if (in_img.empty()) { std::cout << "Error loading the image \n" << std::endl; };
};

cv::Mat NetInput::GetImage() { return in_img; };

std::vector<std::vector<int>> NetInput::getgrid(int ws, int stride) {

    std::vector<std::vector<int>> result;
    for (int y = 0; y <= in_img.cols - ws; y += stride) {
        for (int x = 0; x <= in_img.rows - ws; x += stride) {
            //std::cout << x << y << std::endl;
            result.push_back(std::vector<int> { x, y, x + ws, y + ws });
        }
    }
    return result;
};

cv::Mat NetInput::GetPatch(std::vector<int> bbox) {

    cv::Rect rect(cv::Point(bbox[0], bbox[1]), cv::Point(bbox[2], bbox[3]));
    patch = in_img(rect);
    return patch;
};

/*###########################################################################*/
/*                             UTILITIES                                     */

int findlabel(std::vector<int> orig, std::vector<int> curr, double& overlap, double thr_overlap) {

    double x_intersection = std::max(0, std::min(orig[2], curr[2]) - std::max(orig[0], curr[0]));
    double y_intersection = std::max(0, std::min(orig[3], curr[3]) - std::max(orig[1], curr[1]));
    double intersection_area = x_intersection * y_intersection;
    double orig_area = (orig[2] - orig[0]) * (orig[3] - orig[1]);
    double curr_area = (curr[2] - curr[0]) * (curr[3] - curr[1]);
    double union_area = orig_area + curr_area - intersection_area;
    overlap = intersection_area / orig_area;
    if (intersection_area / union_area > thr_overlap) return 1;
    else return 0;
};

std::vector<int> center_from_bbox(std::vector<int> bbox) {
    int dx = bbox[2] - bbox[0];
    int dy = bbox[3] - bbox[1];
    std::vector<int> center = {bbox[0]+static_cast<int>(dx/2), bbox[1]+ static_cast<int>(dy/2)};
    return center;
};
