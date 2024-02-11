// #include <opencv2/opencv.hpp>
// #include <iostream>

// using namespace cv;
// using namespace std;

// // Function to inpaint and remove black pixels from image edges
// void inpaintEdges(Mat& image, int edgeThickness) {
//     // Create a mask to identify black pixels at the edges
//     Mat mask = Mat::zeros(image.size(), CV_8U);
//     rectangle(mask, Rect(0, 0, image.cols, edgeThickness), Scalar(255), FILLED);  // Top edge
//     rectangle(mask, Rect(0, image.rows - edgeThickness, image.cols, edgeThickness), Scalar(255), FILLED);  // Bottom edge
//     rectangle(mask, Rect(0, 0, edgeThickness, image.rows), Scalar(255), FILLED);  // Left edge
//     rectangle(mask, Rect(image.cols - edgeThickness, 0, edgeThickness, image.rows), Scalar(255), FILLED);  // Right edge

//     // Apply inpaint to remove black pixels from edges
//     inpaint(image, mask, image, 3, INPAINT_TELEA);
// }

// int main(int argc, char** argv) {
//     // Read the input image
//     Mat inputImage = imread("C:\\Users\\akash\\Desktop\\Project3\\x64\\Release\\result.jpg");

//     if (inputImage.empty()) {
//         cout << "Error: Could not read the input image." << endl;
//         return -1;
//     }

//     // Set the edge thickness to be inpainted (you can adjust this based on your requirements)
//     int edgeThickness = 250;

//     // Clone the input image to keep the original intact
//     Mat inpaintedImage = inputImage.clone();

//     // Inpaint to remove black pixels from edges
//     inpaintEdges(inpaintedImage, edgeThickness);

//     // Display the original and inpainted images
//     namedWindow("Original Image", WINDOW_NORMAL);
//     imshow("Original Image", inputImage);

//     namedWindow("Inpainted Image", WINDOW_NORMAL);
//     imshow("Inpainted Image", inpaintedImage);

//     waitKey(0);

//     return 0;
// }


#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to find the farthest non-black pixel from each edge
int calculateEdgeThickness(const Mat& image) {
    int topThickness = 0, bottomThickness = 0, leftThickness = 0, rightThickness = 0;

    // Top edge
    for (int row = 0; row < image.rows; ++row) {
        if (image.at<Vec3b>(row, image.cols / 2) != Vec3b(0, 0, 0)) {
            topThickness = row;
            break;
        }
    }

    // Bottom edge
    for (int row = image.rows - 1; row >= 0; --row) {
        if (image.at<Vec3b>(row, image.cols / 2) != Vec3b(0, 0, 0)) {
            bottomThickness = image.rows - 1 - row;
            break;
        }
    }

    // Left edge
    for (int col = 0; col < image.cols; ++col) {
        if (image.at<Vec3b>(image.rows / 2, col) != Vec3b(0, 0, 0)) {
            leftThickness = col;
            break;
        }
    }

    // Right edge
    for (int col = image.cols - 1; col >= 0; --col) {
        if (image.at<Vec3b>(image.rows / 2, col) != Vec3b(0, 0, 0)) {
            rightThickness = image.cols - 1 - col;
            break;
        }
    }

    // Add some padding (e.g., 100) to ensure inpainting covers a wider area
    int padding = 30;

    return max({ topThickness, bottomThickness, leftThickness, rightThickness }) + padding;
}

// Function to inpaint and remove black pixels from image edges with custom thickness values
void inpaintEdges(Mat& image, int topThickness, int bottomThickness, int leftThickness, int rightThickness) {
    // Create a mask to identify black pixels at the edges
    Mat mask = Mat::zeros(image.size(), CV_8U);

    // Top edge
    if (topThickness > 0)
        rectangle(mask, Rect(0, 0, image.cols, topThickness), Scalar(255), FILLED);

    // Bottom edge
    if (bottomThickness > 0)
        rectangle(mask, Rect(0, image.rows - bottomThickness, image.cols, bottomThickness), Scalar(255), FILLED);

    // Left edge
    if (leftThickness > 0)
        rectangle(mask, Rect(0, 0, leftThickness, image.rows), Scalar(255), FILLED);

    // Right edge
    if (rightThickness > 0)
        rectangle(mask, Rect(image.cols - rightThickness, 0, rightThickness, image.rows), Scalar(255), FILLED);

    // Apply inpaint to remove black pixels from edges
    inpaint(image, mask, image, 3, INPAINT_TELEA);
}

int main(int argc, char** argv) {
    // Read the input image
    Mat inputImage = imread("C:\\Users\\akash\\Desktop\\Project3\\x64\\Release\\panorama002.jpg");

    if (inputImage.empty()) {
        cout << "Error: Could not read the input image." << endl;
        return -1;
    }

    // Clone the input image to keep the original intact
    Mat inpaintedImage = inputImage.clone();

    // Calculate automatic edge thickness
    int autoEdgeThickness = calculateEdgeThickness(inputImage);

    // Inpaint to remove black pixels from edges with automatic edge thickness calculation
    inpaintEdges(inpaintedImage, autoEdgeThickness, autoEdgeThickness, autoEdgeThickness, autoEdgeThickness);

    // Display the original and inpainted images
    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", inputImage);

    namedWindow("Inpainted Image", WINDOW_NORMAL);
    imshow("Inpainted Image", inpaintedImage);

    waitKey(0);

    return 0;
}
