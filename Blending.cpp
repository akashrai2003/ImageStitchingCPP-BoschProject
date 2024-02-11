#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to inpaint and remove black pixels from image edges
void inpaintEdges(Mat& image, int edgeThickness) {
    // Create a mask to identify black pixels at the edges
    Mat mask = Mat::zeros(image.size(), CV_8U);
    rectangle(mask, Rect(0, 0, image.cols, edgeThickness), Scalar(255), FILLED);  // Top edge
    rectangle(mask, Rect(0, image.rows - edgeThickness, image.cols, edgeThickness), Scalar(255), FILLED);  // Bottom edge
    rectangle(mask, Rect(0, 0, edgeThickness, image.rows), Scalar(255), FILLED);  // Left edge
    rectangle(mask, Rect(image.cols - edgeThickness, 0, edgeThickness, image.rows), Scalar(255), FILLED);  // Right edge

    // Apply inpaint to remove black pixels from edges
    inpaint(image, mask, image, 3, INPAINT_TELEA);
}

int main(int argc, char** argv) {
    // Read the input image
    Mat inputImage = imread("C:\\Users\\akash\\Desktop\\Project3\\x64\\Release\\result.jpg");

    if (inputImage.empty()) {
        cout << "Error: Could not read the input image." << endl;
        return -1;
    }

    // Set the edge thickness to be inpainted (you can adjust this based on your requirements)
    int edgeThickness = 250;

    // Clone the input image to keep the original intact
    Mat inpaintedImage = inputImage.clone();

    // Inpaint to remove black pixels from edges
    inpaintEdges(inpaintedImage, edgeThickness);

    // Convert images to the same type for blending
    inpaintedImage.convertTo(inpaintedImage, CV_32F);
    inputImage.convertTo(inputImage, CV_32F);

    // Blend the inpainted image with the original using alpha blending
    float alpha = 0.5;
    Mat blendedImage;
    addWeighted(inpaintedImage, alpha, inputImage, 1 - alpha, 0, blendedImage);

    // Display the original, inpainted, and blended images
    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", inputImage / 255.0);  // Divide by 255 for correct display

    namedWindow("Inpainted Image", WINDOW_NORMAL);
    imshow("Inpainted Image", inpaintedImage / 255.0);  // Divide by 255 for correct display

    namedWindow("Blended Image", WINDOW_NORMAL);
    imshow("Blended Image", blendedImage / 255.0);  // Divide by 255 for correct display

    waitKey(0);

    return 0;
}