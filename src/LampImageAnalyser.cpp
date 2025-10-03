#include "LampImageAnalyser.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/fast_math.hpp>
#include <QDebug>

void LampImageAnalyser::RunEdgeDetection(size_t height, size_t width, uint8_t* buffer, size_t stride)
{
    cv::Mat inputImage;

    try
    {
        inputImage = cv::Mat(height, width, CV_8U, buffer, stride);
        cv::Mat sobelx, sobely, gradient;
 
        // Apply Sobel operator
        cv::Sobel(inputImage, sobelx, CV_64F, 1, 0, 3);
        cv::Sobel(inputImage, sobely, CV_64F, 0, 1, 3);
    
        // Compute gradient magnitude
        cv::magnitude(sobelx, sobely, gradient);
    
        // Convert to 8-bit image
        cv::Mat gradient_abs;
        cv::convertScaleAbs(gradient, gradient_abs);
    
        // Display result
        cv::imshow("Sobel Edge Detection", gradient_abs);

        //emit ImageProcessed(edgedImage);
    }
    catch(const cv::Exception e)
    {
        qDebug() << e.what() << '\n';
    }    
    catch(const std::exception e)
    {
        qDebug() << e.what() << '\n';
    }
    
}

