#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    // List of input image filenames
    vector<String> img_names = {
        R"(C:\Users\DELL\Downloads\SVS_Problem_Statement\Problem_Statement_2\Input_Images\img1.jpg)",
        R"(C:\Users\DELL\Downloads\SVS_Problem_Statement\Problem_Statement_2\Input_Images\img2.jpg)",
        R"(C:\Users\DELL\Downloads\SVS_Problem_Statement\Problem_Statement_2\Input_Images\img3.jpg)",
        R"(C:\Users\DELL\Downloads\SVS_Problem_Statement\Problem_Statement_2\Input_Images\img4.jpg)"
    };

    // Read input images
    vector<Mat> images;
    for (const auto& img_name : img_names) {
        Mat img = imread(img_name);
        if (img.empty()) {
            cerr << "Error: Unable to read image " << img_name << endl;
            return -1;
        }
        images.push_back(img);
    }

    // Set parameters for image stitching
    Stitcher::Mode mode = Stitcher::PANORAMA;
    Ptr<Stitcher> stitcher = Stitcher::create(mode);

    // Perform image stitching
    Mat result;
    Stitcher::Status status = stitcher->stitch(images, result);

    // Check if stitching is successful
    if (status != Stitcher::OK) {
        cerr << "Error: Stitching failed with status code " << status << endl;
        return -1;
    }

    // Save the result
    imwrite("output55.jpg", result);

    cout << "Stitching completed successfully. Output saved as output.jpg" << endl;

    return 0;
}
