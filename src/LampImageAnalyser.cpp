#include "LampImageAnalyser.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/fast_math.hpp>
#include <QDebug>
#include <QFileDialog>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/ximgproc.hpp>
#include "curveAnalysis.hpp"

#include <unordered_set>
#include <algorithm>
#include <numeric>

const int centerApproxTolerance = 30; // px

std::vector<cv::Scalar> coloursRandom = {
    { 231,  72,  28},
    {  57, 203, 155},
    { 142,  89, 221},
    { 255, 183,  64},
    {  90,  96, 242},
    { 214,  40, 123},
    {  67, 214,  52},
    { 156, 240, 210},
    { 255, 112, 176},
    {  48, 169, 255},
    { 255, 255, 102},
    { 116,  61,  34},
    { 185, 122, 222},
    {  33, 222, 188},
    { 255,  77,  77}
};

cv::Scalar colours[] = {
    cv::Scalar(0, 0, 0), 
    cv::Scalar(255, 0, 0), 
    cv::Scalar(255, 0, 255), 
    cv::Scalar(0, 0, 255),
    cv::Scalar(10, 255, 0),
    cv::Scalar(0, 255, 255),
    cv::Scalar(64, 64, 255),
    cv::Scalar(255, 128, 128),
    cv::Scalar(255, 20, 33),
    cv::Scalar(0, 20, 33),
    cv::Scalar(255, 100, 10),
    cv::Scalar(20, 20, 33),
    cv::Scalar(20, 20, 255),
    cv::Scalar(66, 55, 33),
    cv::Scalar(50, 50, 50)
};

void onMouse(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN && userdata)
    {
        auto* imgPtr = static_cast<cv::Mat*>(userdata);

        QString filename = QFileDialog::getSaveFileName(
            nullptr,
            "Save Image As",
            "plasma.png",
            "Images (*.png *.jpg *.bmp);;All Files (*)"
        );

        if (!filename.isEmpty()) {
            if (!cv::imwrite(filename.toStdString(), *imgPtr)) {
                qWarning() << "Failed to save image.";
            } else {
                qDebug() << "Image saved to:" << filename;
            }
        }
    }
}

LampImageAnalyser::LampImageAnalyser()
{
    this->currentImageCv = std::make_shared<cv::Mat>();
    this->originalImageCv = std::make_shared<cv::Mat>();
};

void LampImageAnalyser::SetOriginalImage(const QImage& image)
{
    cv::Mat view(image.height(), image.width(), CV_8UC3,
                 const_cast<uchar*>(image.bits()), image.bytesPerLine());
    *this->originalImageCv = view.clone(); 
}

void LampImageAnalyser::SetCurrentImage(const QImage& image)
{
    cv::Mat view(image.height(), image.width(), CV_8UC3,
                 const_cast<uchar*>(image.bits()), image.bytesPerLine());
    *this->currentImageCv = view.clone(); 
}

void LampImageAnalyser::ResetImage()
{
    *this->currentImageCv = this->originalImageCv->clone();
    this->lamp.ResetLampModel();
    this->fullResultsString = "";
}

cv::Mat pruneSpurs(const cv::Mat& thin255, int iterations = 10) 
{
    CV_Assert(thin255.type() == CV_8U);
    cv::Mat skel; 
    cv::threshold(thin255, skel, 0, 1, cv::THRESH_BINARY); // 0/1

    // 8-neighborhood count (ignore center)
    cv::Mat K = cv::Mat::ones(3,3,CV_8U); 
    K.at<uchar>(1,1) = 0;

    for (int i = 0; i < iterations; ++i) 
    {
        cv::Mat nbrs;
        cv::filter2D(skel, nbrs, CV_8U, K, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
        cv::Mat endpoints = (skel == 1) & (nbrs == 1);  // degree==1
        if (cv::countNonZero(endpoints) == 0) 
        {
            break;
        }
        skel.setTo(0, endpoints);
    }
    skel *= 255;
    return skel;
}

cv::Mat SobelFilter(cv::Mat& inputImg, double threshMultiplier)
{
    cv::Mat imgBlur;
    //cv::blur(imgGray, imgBlur, cv::Size(10,10));
    cv::medianBlur(inputImg, imgBlur, 15);

    cv::Mat normalized;
    cv::normalize(imgBlur, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::threshold(normalized, normalized, 225, 255, cv::THRESH_TRUNC);
    cv::normalize(normalized, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    cv::GaussianBlur(normalized, imgBlur, cv::Size(5,5), 0, 0);

    // SOBEL edge detection
    cv::Mat sobelx, sobely, gradient;
    cv::Sobel(imgBlur, sobelx, CV_64F, 1, 0, 3);
    cv::Sobel(imgBlur, sobely, CV_64F, 0, 1, 3);
    cv::magnitude(sobelx, sobely, gradient);
    cv::Mat gradient_abs;
    cv::convertScaleAbs(gradient, gradient_abs);  

    // make max brightness
    cv::normalize(gradient_abs, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::Mat thresh1;
    cv::threshold(normalized, thresh1, 128, 255, cv::THRESH_TOZERO_INV);
    cv::normalize(thresh1, thresh1, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::imshow("thresh1", thresh1);

    cv::Mat edgesMask;
    // cv::inRange(thresh1, 20, 200, edgesMask);
    // // cv::imshow("edgesMask", edgesMask);
    // cv::morphologyEx(edgesMask, edgesMask, cv::MORPH_OPEN,
    //                 cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));

    // OTSU ALGO
    int otsuthresh = cv::threshold(thresh1, edgesMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU | cv::THRESH_DRYRUN);
    cv::threshold(thresh1, edgesMask, otsuthresh * threshMultiplier, 255, cv::THRESH_BINARY);
            cv::morphologyEx(edgesMask, edgesMask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));
    //cv::imshow("otsuBin", edgesMask);

    return edgesMask;
}

cv::Mat MorphGradDetection(cv::Mat & inputImg, int radius, int thr)
{
    cv::Mat imgBlur;
    //cv::blur(imgGray, imgBlur, cv::Size(10,10));
    cv::medianBlur(inputImg, imgBlur, 11);

    cv::Mat normalized;
    cv::normalize(imgBlur, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::GaussianBlur(normalized, normalized, cv::Size(5, 5), 0);

    cv::Mat se = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(2 * 3 + 1, 2 * radius + 1)
    );

    cv::Mat grad;
    cv::morphologyEx(normalized, grad, cv::MORPH_GRADIENT, se);
    
    cv::normalize(grad, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // // extract a range of brightness
    // cv::Mat thresh;
    // // cv::threshold(normalized, thresh, 75, 75, cv::THRESH_TOZERO_INV);
    // cv::Mat normThresh;
    // cv::normalize(thresh, normThresh, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::imshow("normalized morphgrad", normalized);

    // cv::Mat edgeBin;
    // if (thr > 0) 
    // {
    //     cv::threshold(normalized, edgeBin, thr, 255, cv::THRESH_BINARY);

    //     // Optional cleanup: remove tiny specks and close small gaps
    //     cv::morphologyEx(edgeBin, edgeBin, cv::MORPH_OPEN,
    //                      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
    //     cv::morphologyEx(edgeBin, edgeBin, cv::MORPH_CLOSE,
    //                      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
    // }

    // get rid of "halo" and noisy lines
    // cv::Mat edgesMask;
    // cv::inRange(normalized, 0, 200, edgesMask);


    return normalized;
}

std::vector<std::vector<cv::Point>> GetContours(cv::Mat img, double threshMultiplier)
{
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // normalize to brightness
    cv::Mat edgesMask; 
    edgesMask = SobelFilter(imgGray, threshMultiplier);
    //cv::imshow("sobel result", edgesMask);

    // edgesMask = MorphGradDetection(imgGray, 5, 20);
    // cv::imshow("morph grad result 5", edgesMask);

    // THINNING
    cv::Mat thinEdges;
    cv::ximgproc::thinning(edgesMask, thinEdges, cv::ximgproc::THINNING_ZHANGSUEN);
    //cv::imshow("contoured", thinEdges);
    
    cv::Mat finalEdges = pruneSpurs(thinEdges);
    // TRACING
    std::vector<std::vector<cv::Point>> traced;
    cv::findContours(finalEdges, traced, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::sort(traced.begin(), traced.end(),
            [](auto &a, auto &b)
            { 
            return a.size() > b.size(); 
            }
    );
    
    cv::Mat tracedImg = img.clone();
    for (size_t i = 0; i < traced.size(); ++i)
    {
        if (traced[i].size() < 100)
        {
            continue;
        }
        cv::polylines(tracedImg, traced[i], false, coloursRandom[i]);
    }
    //cv::imshow("traced lamp", tracedImg);

    return traced;
}

bool LampImageAnalyser::ProcessLampWalls(QImage& inputImage)
{
    bool success = true;
    uint8_t * buffer = (uint8_t*)inputImage.bits();
    size_t width = inputImage.width();
    size_t height = inputImage.height();
    size_t stride = inputImage.bytesPerLine();

    try
    {
        cv::Mat img = cv::Mat(height, width, CV_8UC3, buffer, stride);

        this->contours.clear();
        this->contours = GetContours(img, sobelThreshMultiplierDefault);

        if (this->contours.size() < 4)
        {
            success = false;
        }
        else 
        {
            if (this->lamp.lampCenter == this->lamp.undeterminedPoint)
            {
                bool result = this->CalculateElectrodes(inputImage);
                if (result == false)
                {
                    success = false;
                }
            }
            if (success)
            {
                success = this->SimplifyLampContours();
                
                // retry!
                if (!success)
                {
                    double threshMultipliers[] = 
                    {
                        sobelThreshMultiplierDefault - 0.1,
                        sobelThreshMultiplierDefault - 0.15,
                        sobelThreshMultiplierDefault - 0.2,
                        sobelThreshMultiplierDefault - 0.25,
                        sobelThreshMultiplierDefault - 0.3,
                        sobelThreshMultiplierDefault - 0.35,
                        sobelThreshMultiplierDefault - 0.4

                    };
                    size_t tries = sizeof(threshMultipliers)/sizeof(double) - 1;
                    
                    int indexMultiplier = 0;
                    for (int i = 0; i < tries; ++i)
                    {
                        this->contours = GetContours(img, threshMultipliers[indexMultiplier]);
                        success = this->SimplifyLampContours();
                        qDebug() << "retry " << indexMultiplier << "multiplier:" << threshMultipliers[indexMultiplier];
                        if ( success || indexMultiplier == tries)
                        {
                            break;
                        }
                        indexMultiplier ++;
                    }
                }
                
                if (success)
                {
                    std::vector<float> radii;
                    std::vector<QPoint> centers;
                    this->CalculateRadii(&radii, &centers);

                    std::vector<float> radiiMm;
                    for (float radius : radii)
                    {
                        float radiusMm = 0;
                        radiusMm = radius / this->lamp.pixelMmRatio;
                        radiiMm.push_back(radiusMm);
                    }

                    QPair<double, double> lampWalls = this->CalculateWallThickness();
                    this->lamp.wallsCenters = centers;
                    this->lamp.wallsRadii = radiiMm;
                    this->lamp.wallsThickness = lampWalls;
                    
                    //emit ImageProcessed(this->currentImageCv);
                    // emit LampWallsProcessed(centers, radiiMm, lampWalls);
                }
                else
                {

                }
            }
        }
        if (!success)
        {
            emit WallsAnalysisFailed();
        }
    }
    catch(const cv::Exception e)
    {
        qDebug() << e.what() << '\n';
        success = false;
    }    
    catch(const std::exception e)
    {
        qDebug() << e.what() << '\n';
        success = false;
    }
    return success;
}

void LampImageAnalyser::PaintVertice(const QPointF& imgPt)
{
    cv::Mat drawn = this->currentImageCv->clone();

    cv::drawMarker(
        drawn,
        cv::Point(cvRound(imgPt.x()), cvRound(imgPt.y())),
        cv::Scalar(255,255,255),
        cv::MARKER_CROSS,   // <- use a valid marker type
        10,                 // markerSize
        1                   // thickness
    );
    
    cv::String coordinates = cv::String(std::format("({},{})", imgPt.x(), imgPt.y()));
    cv::putText(drawn, 
                coordinates, 
                cv::Point(imgPt.x() + 5, imgPt.y() + 5),
                cv::FONT_HERSHEY_PLAIN,
                0.9,
                cv::Scalar(255, 0, 0)
    );

    *this->currentImageCv = std::move(drawn);

    emit ShowVerticeOnImage(this->currentImageCv);
}

void LampImageAnalyser::PaintVertice(const cv::Point& imgPt)
{
    cv::Mat drawn = this->currentImageCv->clone();

    cv::drawMarker(
        drawn,
        cv::Point(cvRound(imgPt.x), cvRound(imgPt.y)),
        cv::Scalar(255,255,255),
        cv::MARKER_CROSS,
        10,
        2
    );
    
    cv::String coordinates = cv::String(std::format("({},{})", imgPt.x, imgPt.y));
    cv::putText(drawn, 
                coordinates, 
                cv::Point(imgPt.x + 5, imgPt.y + 5),
                cv::FONT_HERSHEY_PLAIN,
                0.9,
                cv::Scalar(255, 0, 0)
    );

    *this->currentImageCv = std::move(drawn);

    emit ShowVerticeOnImage(this->currentImageCv);
}

void LampImageAnalyser::PaintCircleCenter(cv::Point point, int curveNumber, cv::Scalar col)
{
    cv::Mat circleImg = this->currentImageCv->clone();
    cv::circle(circleImg, point, 3, col, cv::FILLED);
    cv::String coordinates = cv::String(std::format("({},{})", point.x, point.y));

    // cv::putText(*this->currentImageCv, 
    //             coordinates, 
    //             cv::Point(point.x + 5, point.y + 5),
    //             cv::FONT_HERSHEY_PLAIN,
    //             1,
    //            col
    //);
    *this->currentImageCv = std::move(circleImg);
    emit ShowVerticeOnImage(this->currentImageCv);
}

static std::vector<cv::Point> resampleByArcLen(const std::vector<cv::Point>& P, float ds, bool closed=false) 
{
    std::vector<cv::Point> Q; 
    Q.reserve(P.size() + 1);

    for (auto& p : P) 
    {
       Q.emplace_back(p);
    }
    if (closed && (Q.front() != Q.back())) 
    {
       Q.push_back(Q.front());
    }

    std::vector<float> S(Q.size(), 0.f);
    for (size_t i = 1; i < Q.size(); ++i) 
    {
        S[i] = S[i - 1] + cv::norm(Q[i] - Q[i -1 ]);
    }
        
    float L = S.back(); 
    if (L <= ds) 
    {
        return Q;
    }

    int M = std::max(2, int(std::floor(L/ds)) + 1);
    std::vector<cv::Point> R; 
    R.reserve(M);
    for (int m = 0; m < M; ++m) 
    {
        float s = std::min(m*ds, L);
        auto it = std::lower_bound(S.begin(), S.end(), s);
        size_t j = std::clamp<size_t>(std::max<int>(1, int(it - S.begin())), 1, Q.size() - 1);
        float t = (S[j] > S[j-1]) ? (s - S[j - 1])/(S[j]-S[j - 1]) : 0.f;
        R.emplace_back(Q[j - 1] + t*(Q[j]-Q[j -1 ]));
    }
    return R;
}
static void drawPolylineF(cv::Mat& img,
                          const std::vector<cv::Point2f>& poly,
                          bool closed,
                          const cv::Scalar& color,
                          int thickness=2,
                          int lineType=cv::LINE_AA)
{
    if (poly.size() < 2) 
    {
        return;
    }

    auto toInt = [](const cv::Point2f& p)
    { 
        return cv::Point(cvRound(p.x), cvRound(p.y)); 
    };

    for (size_t i = 1; i < poly.size(); ++i)
    {
       cv::line(img, toInt(poly[i-1]), toInt(poly[i]), color, thickness, lineType);
    }
    if (closed)
    {
      cv::line(img, toInt(poly.back()), toInt(poly.front()), color, thickness, lineType);
    }
}


typedef enum 
{
    RIGHT,
    LEFT,
    AMBIG
} Orientation;

// to find the curve apex
int findCurveApex(std::vector<cv::Point> &curve, Orientation orientation, int lampCenterY)
{
    // apex at max or min x and approximately at lampCenterY
    int max_x = 0;
    int min_x = 10000;
    int apexIndex = 0;
    int minDiff = centerApproxTolerance;

    if (orientation == Orientation::AMBIG)
    {
        return 0;
    }
    for (size_t i = 0; i < curve.size(); i++)
    {
        int diff = std::abs(lampCenterY - curve[i].y);
        if (diff <= minDiff)
        {
            if (orientation == Orientation::RIGHT)
            {
                if (curve[i].x <= min_x)
                {
                    min_x = curve[i].x;
                    apexIndex = i;
                }
            }
            else if (orientation == Orientation::LEFT)
            {
                if (curve[i].x >= max_x)
                {
                    max_x = curve[i].x;
                    apexIndex = i;
                }
            }
        }
    }
    return apexIndex;
}

// void LampImageAnalyser::FindPixelMmRatio(int lampWidthPx)
// {
//     this->lamp.pixelMmRatio = lampWidthPx / standardLampWidth_mm;
//     qDebug() << "width" << lampWidthPx;
// }

// double CalculateMeanXAxis(const std::vector<cv::Point>* data)
// {
//     size_t size = data->size();
//     if (size == 0)
//     {
//         return 0;
//     }

//     double sum = 0;

//     for (auto point : *data)
//     {
//         sum += point.x;
//     }

//     return sum / size;
// }

// double CalcStdDevXAxis(const std::vector<cv::Point>* data)
// {
//     size_t size = data->size();
//     if (size == 0)
//     {
//         return 0;
//     }
    
//     double mean = CalculateMeanXAxis(data);

//     double sumSquaredDiff = 0;

//     for (cv::Point point : *data)
//     {
//         sumSquaredDiff += std::pow(point.x - mean, 2);
//     }

//     return std::sqrt(sumSquaredDiff / size);
// }

// void RemoveOutliersXAxis(const std::vector<cv::Point>* input, std::vector<cv::Point>* output)
// {
//     double stdDev = CalcStdDevXAxis(input);
//     double mean = CalculateMeanXAxis(input);

//     std::vector<cv::Point> result;

//     for (cv::Point point : *input)
//     {
//         if (std::abs((point.x - mean) / stdDev) < 3)
//         {
//             result.push_back(point);
//         }
//     }
//     *output = result;
// }

static int sign(double v, double eps = 1)
{
    int result = 0;
    if (v > eps)
    {
        result = 1;
    }
    if (v < -eps)
    {
        result = -1;
    }
    return result;
}

void CutCloseToCenter
(
    const std::vector<std::vector<cv::Point>>& input, 
    std::vector<std::vector<cv::Point>> &output, 
    int center, 
    int allowedDistance
)
{
    std::vector<std::vector<cv::Point>> divided;

    for (auto contour : input)
    {
        bool aboveLimit = false;
        size_t pieceStart = 0;
        size_t skipStart  = 0;  

        for (int i = 0; i < contour.size(); ++i)
        {
            bool isOutside = std::abs(contour[i].y - center) > allowedDistance;
            if (!aboveLimit && isOutside)
            {
                // split here
                if (i > pieceStart)
                {
                    divided.emplace_back(contour.begin() + pieceStart, contour.begin() + i);

                }
                aboveLimit = true;
                skipStart = i;
            }
            else if (aboveLimit && !isOutside)
            {
                // cut off points that we skipped
                pieceStart = i;
                aboveLimit = false;
            }
        }
    }
    output = divided;
}

void DivideLongContours(const std::vector<std::vector<cv::Point>>& input, std::vector<std::vector<cv::Point>> &output)
{
    std::vector<std::vector<cv::Point>> divided;

    for (std::vector<cv::Point> curve : input)
    {   
        int derivStep = 5;
        int refineRad = 10;
        int confirmK = 20;
        int n = curve.size();
        std::vector<int> extrema;
        extrema.push_back(0);
        if (n > 3)
        {
            auto slopeSignAt = [&](int i) -> int 
            {
                int a = std::max(0, i - derivStep);
                int b = std::min(n - 1, i + derivStep);
                double s = curve[b].y - curve[a].y;
                return sign(s);
            };

            int i0 = 0;
            int prev = 0;
            // see if the slope changes anywhere along the curve
            while (i0 < n && (prev = slopeSignAt(i0)) == 0) 
            {
                ++i0;
            }
            // reached size, so it is already monotonous
            if (i0 == n) 
            {
                divided.push_back(curve);
                continue;
            }
            
            for (int i = i0 + 1; i < n; ++i) 
            {
                int s = slopeSignAt(i);
                if (s == 0 || s == prev) 
                {
                    continue;
                }

                // Sign flip: + -> - is a local max, - -> + is a local min.
                int a = std::max(0, i - refineRad);
                int b = std::min(n - 1, i + refineRad);

                int best = i;
                if (prev == +1 && s == -1) 
                {
                    // local maximum: refine to max Y
                    double bestY = curve[i].y;
                    for (int k = a; k <= b; ++k) 
                    {
                        if (curve[k].y > bestY) 
                        { 
                            bestY = curve[k].y; 
                            best = k; 
                        }
                    }
                } 
                else if (prev == -1 && s == +1) 
                {
                    // local minimum: refine to min Y
                    double bestY = curve[i].y;
                    for (int k = a; k <= b; ++k) 
                    {
                        if (curve[k].y < bestY) 
                        { 
                            bestY = curve[k].y; 
                            best = k; 
                        }
                    }
                }

                // Avoid duplicates / near-duplicates
                if (std::abs(best - extrema.back()) > 50)
                {
                    auto segmentBegin = extrema.back();
                    extrema.push_back(best);
                    std::vector<cv::Point> segment(curve.begin() + segmentBegin, curve.begin() + extrema.back());
                    if (segment.size() > 100)
                    {
                        divided.push_back(segment);
                    }
                }

                prev = s;
            }
        }
    }
    output = divided;
}

// TODO: larger value of tolerance for larger images
// Find indexes of similar contours
// Returns a vector of pairs where each pair contains indexes of similar contours
std::vector<std::vector<cv::Point>> RemoveSimilarContours
(
    const std::vector<std::vector<cv::Point>>& contours,
    double distanceTolerance = 30.0
)
{
    std::vector<std::vector<cv::Point>> curves = contours;
    std::vector<std::pair<size_t, size_t>> similarPairs;
    std::vector<std::vector<cv::Point>> cleanedContours;
    
    for (size_t i = 0; i < curves.size(); ++i)
    {
        std::sort(curves[i].begin(), curves[i].end(), [](const cv::Point& a, const cv::Point& b) {
            return a.y < b.y;
        });
        std::vector<cv::Point> approx;
        double epsilon = 4.0; // Adjust for more/less smoothing
        cv::approxPolyDP(curves[i], approx, epsilon, false);
        curves[i] = approx;
        // Resample curve with equally distributed points
        curves[i] = resampleByArcLen(curves[i], 2.0f, false);
    }

    for (size_t i = 0; i < curves.size(); ++i)
    {
        for (size_t j = i + 1; j < curves.size(); ++j)
        {

            const auto& contour1 = curves[i];
            const auto& contour2 = curves[j];
            
            if (contour1.empty() || contour2.empty())
            {
                continue;
            }
            
            // Compare spatial extent (bounding boxes)
            cv::Rect bbox1 = cv::boundingRect(contour1);
            cv::Rect bbox2 = cv::boundingRect(contour2);
            
            double centerDist = cv::norm(
                cv::Point2f(bbox1.x + bbox1.width / 2.0f, bbox1.y + bbox1.height / 2.0f) -
                cv::Point2f(bbox2.x + bbox2.width / 2.0f, bbox2.y + bbox2.height / 2.0f)
            );
            
            if (centerDist > distanceTolerance)
            {
                continue;
            }
            
            // Compare average point-to-point distances
            double avgDist = 0.0;
            size_t minSize = std::min(contour1.size(), contour2.size());
            for (size_t k = 0; k < minSize; ++k)
            {
                avgDist += cv::norm(contour1[k] - contour2[k]);
            }
            avgDist /= minSize;
            
            if (avgDist < distanceTolerance)
            {
                similarPairs.push_back({i, j});
            }
        }
    }

    for (int i = 0; i < contours.size(); ++i)
    {
        bool toBeRemoved = false;
        for (auto pair : similarPairs)
        {
            if (pair.second == i)
            {
                toBeRemoved = true;
                break;
            }
        }
        if (!toBeRemoved)
        {
            cleanedContours.push_back(contours[i]);
        }
    }
    
    return cleanedContours;
}

bool LampImageAnalyser::SimplifyLampContours()
{
    bool success = true;
    if (this->lamp.lampCenter == this->lamp.undeterminedPoint)
    {
        success = false;
    }

    cv::Mat trueContoured = this->currentImageCv->clone();
    if (success)
    {
        std::vector<std::vector<cv::Point>> simples;
        std::vector<std::vector<cv::Point>> trueContours;
        int leftmostApexX(0), rightmostApexX(0);

        std::vector<std::vector<cv::Point>> segments;

        // split contours into monotonous segments
        //DivideLongContours(this->contours, segments);

        // cut points far from center (easier than divide into monotonous parts at extrema)
        CutCloseToCenter(this->contours, segments, this->lamp.lampCenter.y, 2.5*this->lamp.pixelMmRatio);

        // remove duplicates
        cv::Mat cleacned = this->currentImageCv->clone();
        std::vector<std::vector<cv::Point>> segmentsCleaned;
        segmentsCleaned = RemoveSimilarContours(segments);

        int i = 0;
        for (auto segment : segmentsCleaned)
        {
            cv::polylines(cleacned, segment, false, colours[i]);
            cv::putText(cleacned, 
                cv::String(std::to_string(i)), 
                segment[0],
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                colours[i]
            );
            i++;
        }
        //cv::imshow("cleaned contours", cleacned);

        for (size_t i = 0; i < segmentsCleaned.size(); i++)
        {
            // filter out contours that do not resemble lamp walls
            // TODO: better method?
            if (segmentsCleaned[i].size() < 50)
            {  
                continue;
            }
            
            std::vector<cv::Point> contour = segmentsCleaned[i];
            cv::polylines(trueContoured, contour, false, colours[i]);
            
            std::sort(contour.begin(), contour.end(), [](const cv::Point& a, const cv::Point& b) {
                return a.y < b.y;
            });
            
            OpeningResult r = detectOpeningLR(contour);
            Orientation orient = (r.opening == OpeningLR::Left ? Orientation::LEFT :
                                    r.opening == OpeningLR::Right ? Orientation::RIGHT : Orientation::AMBIG);

            if (orient == Orientation::AMBIG)
            {
                continue;
            }

            int apexIdx = findCurveApex(contour, orient, this->lamp.lampCenter.y);
            
            this->lamp.wallsApexes.push_back(contour[apexIdx]);

            if (leftmostApexX == 0 && rightmostApexX == 0)
            {
                leftmostApexX = contour[apexIdx].x;
                rightmostApexX = contour[apexIdx].x;
            }
            else if (contour[apexIdx].x < leftmostApexX)
            {
                leftmostApexX = contour[apexIdx].x;
            }
            else if (contour[apexIdx].x > rightmostApexX)
            {
                rightmostApexX = contour[apexIdx].x;
            }

            trueContours.push_back(contour);
            cv::polylines(trueContoured, contour, false, colours[i]);
        }
        //cv::imshow("sorted contours", trueContoured);

        // find pixel to mm ratio from lamp width
        this->lamp.lampWidth = std::abs((rightmostApexX - leftmostApexX)) / this->lamp.pixelMmRatio;
        int pixels2mm = 2 * this->lamp.pixelMmRatio;

        int lampCenterY = this->lamp.lampCenter.y;

        for (size_t i = 0; i < trueContours.size(); i++)
        {
            // find point on curve that lies approximately on the center horizontal line of the lamp
            int indexCenter = 0;
            unsigned bestAbsCenter(centerApproxTolerance), bestIdxCenter(0);
            for (size_t j = 0; j < trueContours[i].size(); ++j)
            {
                int diffCenter = std::abs(trueContours[i][j].y - lampCenterY);
                if (diffCenter < bestAbsCenter)
                {
                    bestAbsCenter = diffCenter;
                    indexCenter = j;
                }
            }
            if (indexCenter == 0)
            {
                continue;
            }
            
            // go up and down 2 mm from the center
            int y_up = trueContours[i][indexCenter].y - pixels2mm;
            int y_down = trueContours[i][indexCenter].y + pixels2mm;

            unsigned bestIdx_up(0), bestIdx_down(0);
            unsigned bestAbs_up(10), bestAbs_down(10);// maximum tolerance to Y difference
            
            for (size_t j = 0; j < trueContours[i].size(); j++)
            {
                if (j <= indexCenter)
                {
                    int d_up = std::abs(trueContours[i][j].y - y_up);
                    if (d_up < bestAbs_up)
                    {
                        bestAbs_up = d_up;
                        bestIdx_up = j;
                    }
                }
                else
                {
                    int d_down = std::abs(trueContours[i][j].y - y_down);
                    if (d_down < bestAbs_down)
                    {
                        bestAbs_down = d_down;
                        bestIdx_down = j;
                    }
                }
            }
            if (bestIdx_down > bestIdx_up)
            {
                std::vector<cv::Point> simple(trueContours[i].begin() + bestIdx_up, trueContours[i].begin() + bestIdx_down);
                simples.push_back(simple);
            }
        }

        if (simples.size() != 4)
        {
            success = false;
        }
        else
        {
            this->lamp.lampContours = simples;

            cv::Mat drawn = this->currentImageCv->clone();
            for (size_t i = 0; i < this->lamp.lampContours.size(); ++i) 
            {
                cv::polylines(drawn, this->lamp.lampContours[i], false, colours[i], 1, cv::LINE_AA);
            }
            //cv::imshow("lamp contours", drawn);
            *this->currentImageCv = std::move(drawn);
        }
        
    }

    return success;
}

inline bool fitCircleArcLSQ
(
    const std::vector<cv::Point>& pts,
    cv::Point2d& center,
    double& radius,
    bool refine = false,
    int refine_iters = 15,
    double huber_delta = 1.5
)
{
    const int n = static_cast<int>(pts.size());
    if (n < 3)
    {
        return false;
    }

    // --- 1) Linear Kåsa fit
    cv::Mat1d M(n, 3), b(n, 1);
    for (int i = 0; i < n; ++i) 
    {
        double x = static_cast<double>(pts[i].x);
        double y = static_cast<double>(pts[i].y);
        M(i, 0) = x;
        M(i, 1) = y;
        M(i, 2) = 1.0;
        b(i, 0) = -(x * x + y * y);
    }

    cv::Mat1d abc;
    if (!cv::solve(M, b, abc, cv::DECOMP_NORMAL | cv::DECOMP_LU))
    {
        return false;
    }

    double A = abc(0, 0);
    double B = abc(1, 0);
    double C = abc(2, 0);

    double cx = -A * 0.5;
    double cy = -B * 0.5;
    double r2 = cx * cx + cy * cy - C;
    if (r2 <= 0.0 || !std::isfinite(r2)) 
    {
        return false;
    }

    double r = std::sqrt(r2);

    if (!refine) 
    {
        center = {cx, cy};
        radius = r;
        return true;
    }

    // --- 2) Geometric refinement (Huber-weighted Gauss–Newton)
    for (int iter = 0; iter < refine_iters; ++iter) 
    {
        double H00 = 0, H01 = 0, H02 = 0;
        double H11 = 0, H12 = 0, H22 = 0;
        double g0 = 0, g1 = 0, g2 = 0;

        int used = 0;
        for (int i = 0; i < n; ++i) 
        {
            double xi = static_cast<double>(pts[i].x);
            double yi = static_cast<double>(pts[i].y);
            double dx = xi - cx;
            double dy = yi - cy;
            double di = std::sqrt(dx * dx + dy * dy);
            if (!(di > 1e-12) || !std::isfinite(di))
            {
                continue;  
            } 

            double ri = di - r;
            double w = 1.0;
            if (std::abs(ri) > huber_delta)
            {
                w = huber_delta / std::abs(ri);
            }

            double j0 = -dx / di;
            double j1 = -dy / di;
            double j2 = -1.0;

            H00 += w * j0 * j0;
            H01 += w * j0 * j1;
            H02 += w * j0 * j2;
            H11 += w * j1 * j1;
            H12 += w * j1 * j2;
            H22 += w * j2 * j2;

            g0 += w * j0 * ri;
            g1 += w * j1 * ri;
            g2 += w * j2 * ri;

            ++used;
        }
        if (used < 3) 
        {
            break;
        }

        cv::Mat1d H(3, 3);
        H(0,0)=H00; H(0,1)=H01; H(0,2)=H02;
        H(1,0)=H01; H(1,1)=H11; H(1,2)=H12;
        H(2,0)=H02; H(2,1)=H12; H(2,2)=H22;

        cv::Mat1d g(3,1); g(0)=g0; g(1)=g1; g(2)=g2;

        // small damping
        H(0,0)+=1e-9; H(1,1)+=1e-9; H(2,2)+=1e-9;

        cv::Mat1d delta;
        if (!cv::solve(H, g, delta, cv::DECOMP_CHOLESKY)) 
        {
            if (!cv::solve(H, g, delta, cv::DECOMP_SVD)) 
            {
                break;
            }
        }

        double d_cx = delta(0);
        double d_cy = delta(1);
        double d_r  = delta(2);

        cx += d_cx;
        cy += d_cy;
        r  += d_r;

        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(r) || r <= 0.0)
        {
            return false;
        }
            
        if (std::sqrt(d_cx*d_cx + d_cy*d_cy) < 1e-9 && std::abs(d_r) < 1e-9)
        {
            break;
        }
    }

    center = {cx, cy};
    radius = r;
    return true;
}

void LampImageAnalyser::CalculateRadii(std::vector<float>* radii, std::vector<QPoint>* centers)
{
    QString radiiString;

    for (size_t i=0; i< this->lamp.lampContours.size(); ++i) 
    {
        //CircleStats stats = findCurvatureRadius(this->lamp.lampContours[i]);
        cv::Point2d center;
        double radius;
        bool success = fitCircleArcLSQ(this->lamp.lampContours[i], center, radius);
        if (!success)
        {
            continue;
        }
        
        cv::Mat drawn = this->currentImageCv->clone();

        qDebug() << "Curve " << i << ": fitR=" << radius << "@ (" << center.x << "," << center.y;

        std::string radiusStr(std::format("({}) R={:.1f} px", i + 1, radius));
        std::string curveNumber(std::format("({})", i + 1));
        radiiString += QString::fromStdString(radiusStr) + "; ";
        
        cv::Point coordinates = cv::Point(this->lamp.lampContours[i][0].x, this->lamp.lampContours[i][0].y - 30);

        cv::putText(*this->currentImageCv, 
                cv::String(curveNumber), 
                coordinates,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                colours[i]
        );

        PaintCircleCenter(cv::Point(center.x, center.y), i + 1, colours[i]);

        radii->push_back(radius);
        centers->push_back(QPoint(center.x, center.y));
    }
    //emit RadiiCalculated(radii);
}

// Signed turn: left turn +, right turn -, range (-180, 180]
double turnAngleDeg(const cv::Point2f& v1, const cv::Point2f& v2) {
    double dot = v1.dot(v2);
    double det = v1.x * v2.y - v1.y * v2.x;     // z-component of 2D cross
    return std::atan2(det, dot) * 180.0 / CV_PI;
}

// Interior angle between segments (0..180)
double interiorAngleDeg(const cv::Point2f& v1, const cv::Point2f& v2) {
    double turn = turnAngleDeg(v1, v2);
    double interior = 180.0 - std::fabs(turn);
    // numeric guard
    if (interior < 0)   interior = 0;
    if (interior > 180) interior = 180;
    return interior;
}

// Helper to compute angle between two vectors
double angleBetween(const cv::Point2f& v1, const cv::Point2f& v2) {
    double dot = v1.dot(v2);
    double det = v1.x * v2.y - v1.y * v2.x;
    double angle = std::atan2(det, dot); // radians
    return angle * 180.0 / CV_PI;        // degrees
}

cv::Point FindMinMaxPoint(const std::vector<cv::Point> & vector, bool isMax, bool isX)
{
    cv::Point result;
    std::vector<cv::Point>::const_iterator it;
    if (isMax)
    {
        it = std::max_element(vector.begin(), vector.end(),
            [isX](const cv::Point& a, const cv::Point& b) {
                return isX ? (a.x < b.x) : (a.y < b.y);
            });
    }
    else 
    {
        it = std::min_element(vector.begin(), vector.end(),
            [isX](const cv::Point& a, const cv::Point& b) {
                return isX ? (a.x < b.x) : (a.y < b.y);
            });
    }
    result = *it;
    return result;
}

ApexSanityResult ApexSanityCheck
(
    const std::vector<cv::Point> &shape,
    cv::Point apex, 
    int imgHeight
)
{
    ApexSanityResult result = ApexSanityResult::APEX_OK;

    if (shape.empty())
    {
        return APEX_ERR_EMPTY_VECTOR;
    }

    // determine orientation

    double dy = static_cast<double>(apex.y) - static_cast<double>(shape.front().y);
    if (dy == 0)
    {
        dy = static_cast<double>(apex.y) - static_cast<double>(shape.back().y);
    }
    if (dy == 0)
    {
        return APEX_ERR_FAULTY_CONTOUR;
    }

    auto findMinMaxYAtX = [](const std::vector<cv::Point>& pts, int x, bool isUpright) -> std::optional<cv::Point> {
        std::optional<cv::Point> best;
        for (const auto& p : pts) {
            if (p.x == x) {
                if ((isUpright && (!best || p.y > best->y)) // base has larger Y coordinates, almost max
                || (!isUpright && (!best || p.y < best->y))) // base has smaller Y coordinates, almost 0
                {
                    best = p;
                }
            }
        }
        return best; // empty if no point with that x found
    };

    // really hard to find the base vertices
    int left_x, right_x;
    left_x = FindMinMaxPoint(shape, false, true).x;
    right_x = FindMinMaxPoint(shape, true, true).x;

    bool isShapeUpright = dy > 0 ? false : true;
    cv::Point left = cv::Point(left_x, findMinMaxYAtX(shape, left_x, isShapeUpright)->y);
    cv::Point right = cv::Point(right_x, findMinMaxYAtX(shape, right_x, isShapeUpright)->y);

    // Base is wide enough?
    cv::Point QR = right - left;
    double L2 = QR.dot(QR);
    if (L2 <= 10) // i know, magic number... 10 pix is actually too thin, baby
    {
        result = ApexSanityResult::APEX_ERR_CONTOUR_TOO_THIN;
    }

    if (result == ApexSanityResult::APEX_OK)
    {
        // apex projects on X axis on the base line?
        if (((apex.x < right.x) && (apex.x < left.x))
            || ((apex.x > right.x) && (apex.x > left.x)))
        {
            result = ApexSanityResult::APEX_ERR_APEX_OUTSIDE;
        }

         if (result == ApexSanityResult::APEX_OK)
        {
            // determine the extremum - the actual apex
            cv::Point extremum;
            if (dy > 0) // apex lower than "leg" => apex at the max y of the vector, electrode upside down
            {
                auto it = std::max_element(shape.begin(), shape.end(),
                            [](const cv::Point& a, const cv::Point& b) {
                                return a.y < b.y;   // compare by Y coordinate
                            });
                extremum = *it;
            }
            else if (dy < 0) // apex higher than "leg" => apex at the min y of the vector, electrode upright
            {
                auto it = std::min_element(shape.begin(), shape.end(),
                            [](const cv::Point& a, const cv::Point& b) {
                                return a.y < b.y;   // compare by Y coordinate
                            });
                extremum = *it;
            }
            
            int amplitude = std::max(std::abs(shape.front().y - extremum.y), std::abs(shape.back().y - extremum.y));
            // apex close to max/min of the polyline ? 
            // choosing between 1/3(imgHeight) or contour amplitude because the contour might be too short after approximation
            double tolerance = std::max(amplitude, static_cast<int>(imgHeight*0.33)) * 0.03; // 3 percent from either amplitude or 1/3(img height)
            if (cv::norm(extremum - apex) > tolerance)
            {
                result = ApexSanityResult::APEX_ERR_APEX_FAR;
            }
        }
    }
    
    return result;
}

bool LampImageAnalyser::ProcessElectrodes(QImage& inputImage)
{
    try
    {
        bool success = this->CalculateElectrodes(inputImage);
        if (success)
        {
            PaintVertice(this->lamp.electrodesApexes.first);
            PaintVertice(this->lamp.electrodesApexes.second);
            // QPoint coordFirst(this->lamp.electrodesApexes.first.x, this->lamp.electrodesApexes.first.y);
            // QPoint coordSecond(this->lamp.electrodesApexes.second.x, this->lamp.electrodesApexes.second.y);
            // double distance = this->lamp.electrodesDistancePx;
            // double angle = this->lamp.electrodesAngle;
            // emit ElectrodesProcessed(coordFirst, coordSecond, distance, angle);
            // cv::circle(coloredEdges, this->lamp.lampCenter, 3, cv::Scalar(0, 0, 255));
            // //qDebug() << "distance" << distance << "angle" << angle;
            // qDebug() << "center" << this->lamp.lampCenter.x << this->lamp.lampCenter.y;
        }
        return success;
    }
    catch(...)
    {
        qDebug() << "Exception while processing lamp electrodes";
        return false;
    }

}

QPair<float, float> LampImageAnalyser::CalculateDistanceAngle(cv::Point start, cv::Point end)
{
    return CalculateDistanceAngle(QPointF(start.x, start.y), QPointF(end.x, end.y));
}

QPair<float, float> LampImageAnalyser::CalculateDistanceAngle(QPointF start, QPointF end)
{
    float distance = sqrt( pow((end.x() - start.x()), 2) +
                           pow((end.y() - start.y()), 2)
                        );
    QLineF line(start, end);
    double angle = std::abs(90.0 - line.angle());
    if (angle > 90) 
    {
        angle = 180.0 - angle;
    }

    return QPair(distance, angle);
}

void JoinSegmentsRecursive(std::vector<std::vector<cv::Point>>& input, std::vector<std::vector<cv::Point>>& output, bool isSorted=false)
{
    std::vector<std::vector<cv::Point>> inputVector = input;
    std::vector<std::vector<cv::Point>> jointVector;
    std::vector<std::vector<cv::Point>> outputVector;
    std::vector<int> alreadyJoined;
    
    // first sort for easier comparison
    if (!isSorted)
    {
        for (auto& segment : inputVector)
        {
            std::sort(
                        segment.begin(), 
                        segment.end(), 
                        [](const cv::Point& a, const cv::Point& b) 
                        {
                            return a.y < b.y;
                        }
                    );
        }
    }

    for (int i = 0; i < inputVector.size(); i++)
    {
        if (std::find(alreadyJoined.begin(), alreadyJoined.end(), i) != alreadyJoined.end())
        {
            continue;
        }

        bool toJoin = false;
        bool duplicate = false;

        float pointsTolerance = 20.0; // px
        for (int j = 0; j < inputVector.size(); j++)
        {
            if (j == i)
            {
                continue;
            }

            float distanceBeginToEnd = sqrt( pow((inputVector[j].front().x - inputVector[i].back().x), 2) +
                                        pow((inputVector[j].front().y - inputVector[i].back().y), 2)
                                    );
            float distanceEndToBegin = sqrt( pow((inputVector[j].back().x - inputVector[i].front().x), 2) +
                            pow((inputVector[j].back().y - inputVector[i].front().y), 2)
                        );

            float distanceBeginToBegin = sqrt( pow((inputVector[j].front().x - inputVector[i].front().x), 2) +
                                        pow((inputVector[j].front().y - inputVector[i].front().y), 2)
                                    );
            float distanceEndToEnd = sqrt( pow((inputVector[j].back().x - inputVector[i].back().x), 2) +
                            pow((inputVector[j].back().y - inputVector[i].back().y), 2)
                        );

            
            if (distanceBeginToBegin < pointsTolerance && distanceEndToEnd < pointsTolerance) // similar
            {
                duplicate = true;
                break;
            }

            std::vector<cv::Point> joined;
            
            if (distanceBeginToEnd < pointsTolerance)
            {
                joined = inputVector[i];
                joined.insert(joined.end(), inputVector[j].begin(), inputVector[j].end());
            }
            else if (distanceEndToBegin < pointsTolerance)
            {
                joined = inputVector[j];
                joined.insert(joined.end(), inputVector[i].begin(), inputVector[i].end());
            }
            else if (distanceBeginToBegin < pointsTolerance)
            {
                std::vector<cv::Point> inputIReverse = inputVector[i];
                std::reverse(inputIReverse.begin(), inputIReverse.end());
                joined = inputIReverse;
                joined.insert(joined.end(), inputVector[j].begin(), inputVector[j].end());
            }
            else if (distanceEndToEnd < pointsTolerance)
            {
                std::vector<cv::Point> inputJReverse = inputVector[j];
                std::reverse(inputJReverse.begin(), inputJReverse.end());
                joined = inputVector[i];
                joined.insert(joined.end(), inputJReverse.begin(), inputJReverse.end());            
            }

            if (joined.size() > 1)
            {
                toJoin = true;
                jointVector.push_back(joined);
                alreadyJoined.push_back(i);
                alreadyJoined.push_back(j);
            }
            
        }
        if (!toJoin && !duplicate)
        {
            jointVector.push_back(inputVector[i]);
        }
    }

    if (alreadyJoined.size() != 0)
    {
        JoinSegmentsRecursive(jointVector, outputVector, true);
        output = outputVector;
    }
    else
    {
        output = jointVector;
    }
}

void RemoveDuplicatePoints(std::vector<cv::Point>& input, std::vector<cv::Point>& output)
{
    std::vector<cv::Point> inputVector = input;
    std::vector<cv::Point> outputVector;
    std::vector<int> toBeRemoved;
    for (int i = 0; i < inputVector.size(); i++)
    {
        if (std::find(toBeRemoved.begin(), toBeRemoved.end(), i) != toBeRemoved.end())
        {
            continue;
        }

        cv::Point first = inputVector[i];
        float pointsTolerance = 20.0; // px

        for (int j = 0; j < inputVector.size(); j++)
        {
            if (i == j)
            {
                continue;
            }

            cv::Point second = inputVector[j];

            float distance= sqrt( pow((first.x - second.x), 2) +
                                        pow((first.y - second.y), 2)
                                    );
            if (distance < pointsTolerance)
            {
                toBeRemoved.push_back(j);
            }
        }
    }

    for (int i = 0; i < inputVector.size(); i++)
    {
        if (std::find(toBeRemoved.begin(), toBeRemoved.end(), i) != toBeRemoved.end())
        {
            continue;
        }
        else
        {
            outputVector.push_back(inputVector[i]);
        }
    }
    output = outputVector;
}

bool LampImageAnalyser::CalculateElectrodes(QImage& inputImage)
{
    bool success = true;

    uint8_t * buffer = (uint8_t*)inputImage.bits();
    size_t width = inputImage.width();
    size_t height = inputImage.height();
    size_t stride = inputImage.bytesPerLine();

    cv::Mat img = cv::Mat(height, width, CV_8UC3, buffer, stride);

    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    
    cv::Mat imgBlur;
    cv::GaussianBlur(imgGray, imgBlur, cv::Size(3,3), 1.7);

    // TRACING
    cv::Mat toTrace = img.clone();
    std::vector<std::vector<cv::Point>> electrodesTemp;

    cv::Ptr<cv::ximgproc::FastLineDetector> fld =
        cv::ximgproc::createFastLineDetector(/*length_threshold=*/40,
                                            /*distance_threshold=*/1.4142,
                                            /*canny_th1=*/25, /*canny_th2=*/75,
                                            /*aperture=*/3, /*do_merge=*/true);

    std::vector<cv::Vec4f> lines;
    fld->detect(imgGray, lines);

    std::vector<std::vector<cv::Point>> segments;
    for (const auto& l : lines) 
    {
        segments.push_back({ cv::Point(cvRound(l[0]), cvRound(l[1])),
                            cv::Point(cvRound(l[2]), cvRound(l[3])) });
    }

    // for (auto poly : segments)
    // {
    //     cv::polylines(toTrace, poly, 0, colours[1]);
    // }

    // cv::imshow("raw", toTrace);
    // join broken contours
    this->lamp.electrodes.clear();
    std::vector<std::vector<cv::Point>> electrodesJoint;
    JoinSegmentsRecursive(segments, electrodesJoint);

    // int i = 0;
    // for (auto poly : electrodesJoint)
    // {
    //     cv::polylines(toTrace, poly, 0, colours[i]);
    //     i++;
    // }
    // cv::imshow("joint", toTrace);

    // find vertices
    std::vector<cv::Point> apexesVect;
    ApexSanityResult apexError = ApexSanityResult::APEX_OK;

    for (auto electrode : electrodesJoint)
    {
        if (electrode.size() <= 2)
        {
            // too small -> not connected to anything
            continue;
        }

        this->lamp.electrodes.push_back(electrode);
        // Approximate polyline with straight segments
        std::vector<cv::Point> approx;
        double epsilon = 10.0; // Approximation accuracy
        cv::approxPolyDP(electrode, approx, epsilon, false);

        // Find angles between consecutive segments
        double smallestAngle = 45;
        cv::Point apex(0,0);

        for (size_t i = 1; i + 1 < approx.size(); ++i) {
            // Build direction vectors along the two adjacent segments
            cv::Point2f v1 = cv::Point2f(approx[i].x - approx[i-1].x,
                                        approx[i].y - approx[i-1].y);
            cv::Point2f v2 = cv::Point2f(approx[i+1].x - approx[i].x,
                                        approx[i+1].y - approx[i].y);

            // Optional: ignore tiny/degenerate segments to avoid noisy angles
            if (cv::norm(v1) < 1e-6 || cv::norm(v2) < 1e-6) continue;

            double interior_deg = interiorAngleDeg(v1, v2);    // [0..180]
            if (interior_deg < smallestAngle && interior_deg > 5.0)
            {  
                smallestAngle = interior_deg;
                apex = approx[i];

            }
        }
        // the apex is not sharp enough
        if (apex.x == 0 && apex.y == 0)
        {
            continue;
        }

        // sanity check

        ApexSanityResult isApexOk = ApexSanityCheck(approx, apex, imgGray.rows);

        if (isApexOk == ApexSanityResult::APEX_OK)
        {
            apexesVect.push_back(apex);
        }
        else
        {
            apexError = isApexOk;
            break;
            }
    }

    // might be duplicate points because contours might be duplicated
    if (apexesVect.size() > 2)
    {
        RemoveDuplicatePoints(apexesVect, apexesVect);
    }

    if (apexesVect.empty() || apexesVect.size() != 2 || apexError != ApexSanityResult::APEX_OK)
    {
        emit ElectrodesAnalysisFailed();
        success = false;
    }
    else
    {
        QPair<cv::Point, cv::Point> apexes;
        apexes.first = apexesVect[0];
        apexes.second = apexesVect[1];
        
        auto [distance, angle] = CalculateDistanceAngle(apexes.first, apexes.second);
        this->lamp.lampCenter = cv::Point2f((apexes.first.x + apexes.second.x)/2, (apexes.first.y + apexes.second.y)/2);

        this->lamp.electrodesDistancePx = distance;
        this->lamp.electrodesApexes = apexes;
        this->lamp.electrodesAngle = angle;
    }
    return success;
}

QPair<double, double>  LampImageAnalyser::CalculateWallThickness()
{
    QPair<double, double> result = QPair<size_t, size_t>(0, 0);

    if (!(this->lamp.wallsApexes.empty()))
    {
        std::vector<cv::Point> edgesLeft, edgesRight;
        for (cv::Point apex : this->lamp.wallsApexes)
        {
            if (std::abs(this->lamp.lampCenter.y - apex.y) > centerApproxTolerance)
            {
                continue;
            }
            if (apex.x < this->lamp.lampCenter.x && edgesLeft.size() < 2)
            {
                edgesLeft.push_back(apex);
            }
            else if (apex.x > this->lamp.lampCenter.x && edgesRight.size() < 2)
            {
                edgesRight.push_back(apex);
            }
        }

        if (edgesLeft.size() == 2 && edgesRight.size() == 2)
        {
            auto [wallLeftPx, angleLeft] = CalculateDistanceAngle(edgesLeft[0], edgesLeft[1]);
            auto [wallRightPx, angleRight] = CalculateDistanceAngle(edgesRight[0], edgesRight[1]);

            double ratio = this->lamp.pixelMmRatio;

            double wallLeftMm = ratio != 0 ? wallLeftPx / this->lamp.pixelMmRatio : 0;
            double wallRightMm = ratio != 0 ? wallRightPx / this->lamp.pixelMmRatio : 0;
            result.first = wallLeftMm;
            result.second = wallRightMm;
        }
    }
    return result;
}

void LampImageAnalyser::RunFullAnalysis(QImage& imgToProcess)
{
    bool success = true;
    this->lamp.ResetLampModel();
    this->fullResultsString = "";
    success = this->ProcessElectrodes(imgToProcess);
    if (success)
    {
        success = this->ProcessLampWalls(imgToProcess);

        if (success)
        {
            double ratio = this->lamp.pixelMmRatio;
            double electrodesDistanceMm = ratio != 0 ? this->lamp.electrodesDistancePx / ratio : 0;
            QPoint coordElectrodeFirst(this->lamp.electrodesApexes.first.x, this->lamp.electrodesApexes.first.y);
            QPoint coordElectrodeSecond(this->lamp.electrodesApexes.second.x, this->lamp.electrodesApexes.second.y);
            double electrodesAngle = this->lamp.electrodesAngle;
            
            const QChar delimeter = ',';
            for (auto wallCenter : this->lamp.wallsCenters)
            {
                fullResultsString += QString("(%1;%2)").arg(wallCenter.x()).arg(wallCenter.y()) + delimeter;
            }

            for (auto wallRadius : this->lamp.wallsRadii)
            {
                fullResultsString += QString::number(wallRadius, 'f', 2) + delimeter;
            }

            fullResultsString += QString::number(electrodesDistanceMm, 'f', 2) + delimeter;
            fullResultsString += QString::number(electrodesAngle, 'f', 2) + delimeter;

            fullResultsString += QString::number(this->lamp.wallsThickness.first, 'f', 2) + delimeter
                                + QString::number(this->lamp.wallsThickness.second, 'f', 2) + delimeter;
            
            fullResultsString += QString::number(this->lamp.lampWidth, 'f', 2) + "\r\n";

            emit ElectrodesProcessed(coordElectrodeFirst, coordElectrodeSecond, electrodesDistanceMm, electrodesAngle);
            emit LampWallsProcessed(this->lamp.wallsCenters, this->lamp.wallsRadii, this->lamp.wallsThickness);

            emit ImageProcessed(this->currentImageCv);
        }
    }
}

QString LampImageAnalyser::GetResultsString()
{
    return this->fullResultsString;
}

QString LampImageAnalyser::GetContoursTable()
{
    QString table = "wall 1 x\twall 1 y\twall2 x\twall2 y\twall3 x\twall 3 y\twall4 x\twall4 y\r\n";

    int maxContour = 0;
    for (auto contour : this->lamp.lampContours)
    {
        if (contour.size() > maxContour)
        {
            maxContour = contour.size();
        }
    }

    for (int i = 0; i < maxContour; ++i)
    {
        for (auto contour : this->lamp.lampContours)
        {
            if (contour.size() > i)
            {
                table += QString("%1\t%2\t").arg(contour[i].x).arg(contour[i].y);
            }
            else
            {
                table += ".\t.\t";
            }
        }
        table += "\r\n";

    }
    return table;
}