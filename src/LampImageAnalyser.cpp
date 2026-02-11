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


static double dot(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x*b.x + a.y*b.y;
}

static double cross2D(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x*b.y - a.y*b.x;
}

static double norm(const cv::Point2f& v) {
    return std::sqrt(dot(v, v));
}

static double distance(const cv::Point & a, const cv::Point & b)
{
    cv::Point d = a - b;
    return std::sqrt(d.x*d.x + d.y*d.y);
}

void JoinSegmentsRecursive
(
    std::vector<std::vector<cv::Point>>& input, 
    std::vector<std::vector<cv::Point>>& output, 
    double toleranceLen,
    bool isFirstRun=true,
    ImageType imgType = ImageType::XRAY,
    cv::Point imgDim = cv::Point(2500, 2500)
);

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

cv::Mat SobelFilter(const cv::Mat& inputImg, double threshMultiplier, ImageType type)
{
    cv::Mat imgBlur;
    //cv::blur(imgGray, imgBlur, cv::Size(10,10));
    if (type != ImageType::PHOTO)
    {
        cv::medianBlur(inputImg, imgBlur, 15);
    }
    else
    {
        cv::bilateralFilter(inputImg, imgBlur, 7, 75, 75);
    }

    cv::Mat normalized;
    cv::normalize(imgBlur, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    if (type == ImageType::XRAY)
    {
        cv::threshold(normalized, normalized, 225, 255, cv::THRESH_TRUNC);
        cv::normalize(normalized, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        
        cv::GaussianBlur(normalized, normalized, cv::Size(5,5), 0, 0);
    }

    // SOBEL edge detection
    cv::Mat sobelx, sobely, gradient;
    cv::Sobel(normalized, sobelx, CV_64F, 1, 0, 3);
    cv::Sobel(normalized, sobely, CV_64F, 0, 1, 3); 
    cv::magnitude(sobelx, sobely, gradient);
    cv::Mat gradient_abs;
    cv::convertScaleAbs(gradient, gradient_abs);  

    // make max brightness
    cv::normalize(gradient_abs, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::Mat thresh1;
    if (type == ImageType::XRAY)
    {
        cv::threshold(normalized, thresh1, 128, 255, cv::THRESH_TOZERO_INV);
        cv::normalize(thresh1, thresh1, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    }
    else
    {
        thresh1 = normalized;
    }
    //cv::imshow("thresh1", thresh1);

    cv::Mat edgesMask;

    // OTSU ALGO
    int otsuthresh = cv::threshold(thresh1, edgesMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU | cv::THRESH_DRYRUN);
    cv::threshold(thresh1, edgesMask, otsuthresh * threshMultiplier, 255, cv::THRESH_BINARY);
    // if (type != ImageType::PHOTO)
    // {
        cv::morphologyEx(edgesMask, edgesMask, cv::MORPH_CLOSE,
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7)));
    // }
    // else
    // {
    //     cv::morphologyEx(edgesMask, edgesMask, cv::MORPH_CLOSE,
    //         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
        
    // }
    // cv::imshow("otsuBin", edgesMask);

    return edgesMask;
}

std::vector<std::vector<cv::Point>> GetContours(cv::Mat img, double threshMultiplier, ImageType type)
{
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // normalize to brightness
    cv::Mat edgesMask; 
    edgesMask = SobelFilter(imgGray, threshMultiplier, type);
    //cv::imshow("sobel result", edgesMask);

    // THINNING
    cv::Mat thinEdges;
    cv::ximgproc::thinning(edgesMask, thinEdges, cv::ximgproc::THINNING_ZHANGSUEN);
    //cv::imshow("contoured", thinEdges);
    
    cv::Mat finalEdges = pruneSpurs(thinEdges);
    // TRACING
    std::vector<std::vector<cv::Point>> traced;
    cv::findContours(finalEdges, traced, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    

    std::vector<std::vector<cv::Point>> tracedCleanedUp;

    for (size_t i = 0; i < traced.size(); ++i)
    {
        if (traced[i].size() > 100)
        {
            tracedCleanedUp.push_back(traced[i]);
        }
    }
    // cv::Mat tracedImg = img.clone();
    // for (size_t i = 0; i < tracedCleanedUp.size(); ++i)
    // {
    //     cv::polylines(tracedImg, tracedCleanedUp[i], false, coloursRandom[i]);
    //     double fontScale = 1;
    //     cv::putText(tracedImg, 
    //         cv::String(std::to_string(i)), 
    //         tracedCleanedUp[i][0],
    //         cv::FONT_HERSHEY_DUPLEX,
    //         fontScale,
    //         coloursRandom[i]
    //     );
    // }
    // cv::imshow("traced lamp", tracedImg);

    return tracedCleanedUp;
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
        this->contours = GetContours(img, sobelThreshMultiplierDefault, this->imageType);


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
            
            //retry!
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
                    sobelThreshMultiplierDefault - 0.4,
                    sobelThreshMultiplierDefault - 0.45

                };
                size_t tries = sizeof(threshMultipliers)/sizeof(double);
                
                int indexMultiplier = 0;
                for (int i = 0; i < tries; ++i)
                {
                    this->contours = GetContours(img, threshMultipliers[indexMultiplier], imageType);
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
        cv::Scalar(0,0,0),
        cv::MARKER_CROSS,   // <- use a valid marker type
        10,                 // markerSize
        1                   // thickness
    );
    
    double fontScale = this->currentImageCv->cols / 700;
    cv::String coordinates = cv::String(std::format("({},{})", imgPt.x(), imgPt.y()));
    // cv::putText(drawn, 
    //             coordinates, 
    //             cv::Point(imgPt.x() + 5, imgPt.y() + 5),
    //             cv::FONT_HERSHEY_PLAIN,
    //             fontScale,
    //             cv::Scalar(255, 0, 0)
    // );

    *this->currentImageCv = std::move(drawn);

    emit ShowVerticeOnImage(this->currentImageCv);
}

void LampImageAnalyser::PaintVertice(const cv::Point& imgPt)
{
    cv::Mat drawn = this->currentImageCv->clone();

    cv::drawMarker(
        drawn,
        cv::Point(cvRound(imgPt.x), cvRound(imgPt.y)),
        cv::Scalar(0,0,0),
        cv::MARKER_CROSS,
        10,
        2
    );
    
    cv::String coordinates = cv::String(std::format("({},{})", imgPt.x, imgPt.y));
    double fontScale = this->currentImageCv->cols / 700;
    // cv::putText(drawn, 
    //             coordinates, 
    //             cv::Point(imgPt.x + 5, imgPt.y + 5),
    //             cv::FONT_HERSHEY_DUPLEX,
    //             fontScale,
    //             cv::Scalar(255, 0, 0)
    // );

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
    UP,
    DOWN,
    AMBIG
} Orientation;

// to find the curve apex
int findCurveApex(std::vector<cv::Point> &curve, bool isVertical, cv::Point lampCenter, double tolerance)
{
    // apex at max or min x and approximately at lampCenterY
    int right_x = 0;
    int left_x = 10000;    

    int bottom_y = 0;
    int top_y = 10000;

    int apexIndex = 0;
    int minDiffCenterLine = tolerance;

    for (size_t i = 0; i < curve.size(); i++)
    {
        int diffFromCenterLine;
        if (isVertical) 
        {
            diffFromCenterLine = std::abs(lampCenter.y - curve[i].y);
        }
        else 
        {
            diffFromCenterLine = std::abs(lampCenter.x - curve[i].x);
        }

        if (diffFromCenterLine <= minDiffCenterLine)
        {
            bool isLeftTop = isVertical ? curve[i].x < lampCenter.x
                                        : curve[i].y < lampCenter.y;
        
            if (isLeftTop)
            {
                if (isVertical)
                {
                    if (curve[i].x <= left_x)
                    {
                        left_x = curve[i].x;
                        apexIndex = i;
                    }
                }
                else
                {
                    if (curve[i].y <= top_y)
                    {
                        top_y = curve[i].y;
                        apexIndex = i;
                    }
                }
            }
            else
            {
                if (isVertical)
                {
                    if (curve[i].x >= right_x)
                    {
                        right_x = curve[i].x;
                        apexIndex = i;
                    }
                }
                else
                {
                    if (curve[i].y >= bottom_y)
                    {
                        bottom_y = curve[i].y;
                        apexIndex = i;
                    }
                }
            }
        }
    }
    return apexIndex;
}

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
    cv::Point center, 
    int allowedDistance,
    bool isVertical
)
{
    std::vector<std::vector<cv::Point>> pieces;

    for (auto contour : input)
    {
        if (contour.size() < 100)
        {
            continue;
        }

        std::vector<cv::Point> piece;
        for (int i = 0; i < contour.size(); ++i)
        {
            int dist = isVertical ? contour[i].y - center.y: contour[i].x - center.x;
            if (std::abs(dist) <= allowedDistance)
            {
                piece.push_back(contour[i]);
            }
            else
            {
                if (piece.size() > 10)
                {
                    pieces.push_back(piece);
                    piece.clear();
                }
            }
        }
        if (piece.size() > 10)
        {
            pieces.push_back(piece);
            piece.clear();
        }
    }
    output = pieces;
}

// TODO: larger value of tolerance for larger images
// Find indexes of similar contours
// Returns a vector of pairs where each pair contains indexes of similar contours
std::vector<std::vector<cv::Point>> RemoveSimilarContours
(
    const std::vector<std::vector<cv::Point>>& contours
)
{
    std::vector<std::vector<cv::Point>> curves = contours;
    std::vector<std::pair<size_t, size_t>> similarPairs;
    std::vector<std::vector<cv::Point>> cleanedContours;
    
    double distanceTolerance = 10;

    for (size_t i = 0; i < curves.size(); ++i)
    {
        std::vector<cv::Point> approx;
        double epsilon = 4.0; // Adjust for more/less smoothing
        cv::approxPolyDP(curves[i], approx, epsilon, false);
        // Resample curve with equally distributed points
        curves[i] = resampleByArcLen(approx, 2.0f, false);
    }

    std::vector<bool> markedToRemove(curves.size(), false);

    for (size_t i = 0; i < curves.size(); ++i)
    {
        if (markedToRemove[i])
        {
            continue;
        }
        double len_i = cv::arcLength(curves[i], false);
        for (size_t j = i + 1; j < curves.size(); ++j)
        {
            distanceTolerance = std::min(cv::arcLength(curves[j], false), len_i) * 0.1;

            if (markedToRemove[j])
            {
                continue;
            }
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
                markedToRemove[j] = true;
            }
        }
    }

    for (int i = 0; i < contours.size(); ++i)
    {
        if (!markedToRemove[i])
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

        int startApex(0), endApex(0);

        std::vector<std::vector<cv::Point>> segmentsRaw;

        bool isVertical = this->lamp.isVertical;

        // cut points far from center (easier than divide into monotonous parts at extrema)
        CutCloseToCenter
        (
            this->contours, 
            segmentsRaw, 
            this->lamp.lampCenter, 
            2.5*this->lamp.electrodesDistancePx, 
            isVertical
        );

        int i = 0;
        cv::Mat cut = this->currentImageCv->clone();
        for (auto segment : segmentsRaw)
        {
            cv::polylines(cut, segment, false, colours[i], 3);
            cv::putText(cut, 
                cv::String(std::to_string(i)), 
                segment[0],
                cv::FONT_HERSHEY_DUPLEX,
                1,
                colours[i]
            );
            i++;
        }
        cv::imshow("cut contours", cut);

        for (size_t i = 0; i < segmentsRaw.size(); i++)
        {
            // filter out contours that do not resemble lamp walls
            // TODO: better method?
            if (segmentsRaw[i].size() < 50)
            {  
                continue;
            }
            
            std::vector<cv::Point> contour = segmentsRaw[i];

            std::sort(contour.begin(), contour.end(), [isVertical](const cv::Point& a, const cv::Point& b) {
                return (isVertical ? a.y < b.y : a.x < b.x);
            });
            
            Orientation orient;

            if (isVertical)
            { 
                OpeningResult r;
                r = detectOpeningLR(contour);
                orient = (r.opening == OpeningLR::Left ? Orientation::LEFT :
                                        r.opening == OpeningLR::Right ? Orientation::RIGHT : Orientation::AMBIG);
            }
            else
            {
                OpeningResultUD r;
                r = detectOpeningUD(contour);
                orient = (r.opening == OpeningUD::Up ? Orientation::UP :
                                        r.opening == OpeningUD::Down ? Orientation::DOWN : Orientation::AMBIG);
            }

            if (orient == Orientation::AMBIG)
            {
                continue;
            }

            trueContours.push_back(contour);
        }

        //remove duplicates
        trueContours = RemoveSimilarContours(trueContours);

        i = 0;
        cv::Mat trueContoured = this->currentImageCv->clone();
        for (auto segment : trueContours)
        {
            cv::polylines(trueContoured, segment, false, colours[i], 3);
            cv::putText(trueContoured, 
                cv::String(std::to_string(i)), 
                segment[0],
                cv::FONT_HERSHEY_DUPLEX,
                1,
                colours[i]
            );
            i++;
        }
        cv::imshow("true contours removed", trueContoured);

        int pixels2mm = 2 * this->lamp.pixelMmRatio;

        for (size_t i = 0; i < trueContours.size(); i++)
        {
            // find point on curve that lies approximately on the center horizontal line of the lamp
            int indexCenter = 0;
            double centerApproxTolerance = this->lamp.electrodesDistancePx * 0.5;
            unsigned bestAbsCenter(centerApproxTolerance), bestIdxCenter(0);
            for (size_t j = 0; j < trueContours[i].size(); ++j)
            {
                int diffCenter = isVertical ? std::abs(trueContours[i][j].y - this->lamp.lampCenter.y)
                                            : std::abs(trueContours[i][j].x - this->lamp.lampCenter.x);

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
            
            // go up and down 2 mm from the center (or left and right)
            int y_up = trueContours[i][indexCenter].y - pixels2mm;
            int y_down = trueContours[i][indexCenter].y + pixels2mm;
            int xLeft = trueContours[i][indexCenter].x - pixels2mm;
            int xRight = trueContours[i][indexCenter].x + pixels2mm;

            int bestIdx_up(-1), bestIdx_down(-1);
            // maximum tolerance to Y difference
            unsigned bestAbs_up(this->lamp.pixelMmRatio * 0.5), bestAbs_down(this->lamp.pixelMmRatio*0.5);
            
            for (size_t j = 0; j < trueContours[i].size(); j++)
            {
                if (j <= indexCenter)
                {
                    int d_up = isVertical ? std::abs(trueContours[i][j].y - y_up) 
                                            : std::abs(trueContours[i][j].x - xLeft) ;
                    if (d_up < bestAbs_up)
                    {
                        bestAbs_up = d_up;
                        bestIdx_up = j;
                    }
                }
                else
                {
                    int d_down = isVertical ? std::abs(trueContours[i][j].y - y_down)
                                                : std::abs(trueContours[i][j].x - xRight) ;
                    if (d_down < bestAbs_down)
                    {
                        bestAbs_down = d_down;
                        bestIdx_down = j;
                    }
                }
            }
            if (bestIdx_down > bestIdx_up && bestIdx_up >= 0 && bestIdx_down > 0)
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
            for (auto contour : simples)
            {
                double centerApproxTolerance = this->lamp.electrodesDistancePx * 0.5;
                int apexIdx = findCurveApex(contour, this->lamp.isVertical, this->lamp.lampCenter, centerApproxTolerance);
                
                this->lamp.wallsApexes.push_back(contour[apexIdx]);

                int checkApex = isVertical ? contour[apexIdx].x : contour[apexIdx].y;
                if (startApex == 0 && endApex == 0)
                {
                    startApex = checkApex;
                    endApex = checkApex;
                }
                else if (checkApex < startApex)
                {
                    startApex = isVertical ? checkApex : contour[apexIdx].y;
                }
                else if (checkApex > endApex)
                {
                    endApex = isVertical ? checkApex : contour[apexIdx].y;
                }
            }

            this->lamp.lampWidth = std::abs((startApex - endApex)) / this->lamp.pixelMmRatio;
            this->lamp.lampContours = simples;

            cv::Mat drawn = this->currentImageCv->clone();
            for (size_t i = 0; i < this->lamp.lampContours.size(); ++i) 
            {
                cv::polylines(drawn, this->lamp.lampContours[i], false, colours[i], 3, cv::LINE_AA);
            }
            // cv::imshow("lamp contours", drawn);
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

        // qDebug() << "Curve " << i << ": fitR=" << radius << "@ (" << center.x << "," << center.y;

        // std::string radiusStr(std::format("({}) R={:.1f} px", i + 1, radius));
        // std::string curveNumber(std::format("({})", i + 1));
        // radiiString += QString::fromStdString(radiusStr) + "; ";
        
        //cv::Point coordinates = cv::Point(this->lamp.lampContours[i][0].x, this->lamp.lampContours[i][0].y - 30);

        // double fontScale = this->currentImageCv->cols / 1000;
        // cv::putText(*this->currentImageCv, 
        //         cv::String(curveNumber), 
        //         coordinates,
        //         cv::FONT_HERSHEY_DUPLEX,
        //         fontScale,
        //         colours[i]
        // );

        PaintCircleCenter(cv::Point(center.x, center.y), i + 1, colours[i]);

        radii->push_back(radius);
        centers->push_back(QPoint(center.x, center.y));
    }
    //emit RadiiCalculated(radii);
}

ApexSanityResult ApexSanityCheck
(
    const std::vector<cv::Point> &shape,
    int apexIdx, 
    int imgHeight,
    int imgWidth
)
{
    ApexSanityResult result = ApexSanityResult::APEX_OK;

    if (shape.empty())
    {
        return APEX_ERR_EMPTY_VECTOR;
    }

    bool isVertical = false;

    // really hard to find the base vertices, let's pretend it is a triangle
    cv::Point left = apexIdx > 0 ? shape[apexIdx-1] : cv::Point(0, 0);
    cv::Point right = apexIdx < (shape.size() - 1) ? shape[apexIdx + 1] : cv::Point(0, 0);

    int verticalDiff = std::max(std::abs(shape[apexIdx].y - left.y), std::abs(shape[apexIdx].y - right.y));
    int horizontalDiff = std::max(std::abs(shape[apexIdx].x - left.x), std::abs(shape[apexIdx].x - right.x));
    if (horizontalDiff < verticalDiff)
    {
        isVertical = true;
    }


    // apex projects on X axis on the base line?
    if (isVertical)
    {
        if (((shape[apexIdx].x <= right.x) && (shape[apexIdx].x <= left.x))
            || ((shape[apexIdx].x >= right.x) && (shape[apexIdx].x >= left.x)))
        {
            result = ApexSanityResult::APEX_ERR_APEX_OUTSIDE;
        }
    }
    else
    {
        if (((shape[apexIdx].y <= right.y) && (shape[apexIdx].y <= left.y))
            || ((shape[apexIdx].y >= right.y) && (shape[apexIdx].y >= left.y)))
        {
            result = ApexSanityResult::APEX_ERR_APEX_OUTSIDE;
        }
    }
    

    if (result == ApexSanityResult::APEX_OK)
    {
        // check that the height falling from the apex to the center of the base if almost vertical or almost horizontal
        cv::Point center = cv::Point((left.x + right.x)/2, (left.y + right.y)/2);
        cv::Point v = cv::Point(center - shape[apexIdx]);
        if (v.x == 0 && v.y == 0)
        {
            result = APEX_ERR_FAULTY_CONTOUR;
        }
        else
        {
            double angle = 180.0;
            if (isVertical)
            {
                angle = std::atan2(std::abs(v.x), std::abs(v.y)) * 180.0 / CV_PI;
            }
            else
            {
                angle = std::atan2(std::abs(v.y), std::abs(v.x)) * 180.0 / CV_PI;
            }

            if (angle > 30)
            {
                result =  APEX_ERR_FAULTY_CONTOUR;
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

double CalcMaxLenSector(std::vector<cv::Point> first, std::vector<cv::Point> second, bool isFirstFront, bool isSecondFront)
{
    std::vector<cv::Point> sector1, sector2;
    if (!isFirstFront && isSecondFront)
    {
        sector1 = std::vector<cv::Point>(second.begin(), second.begin() + 2);
        sector2 = std::vector<cv::Point>(first.end() - 2, first.end());
    }                
    else if (isFirstFront && !isSecondFront)
    {
        sector1 = std::vector<cv::Point>(second.end() - 2, second.end());
        sector2 = std::vector<cv::Point>(first.begin(), first.begin() + 2);
    }
    else if (isFirstFront && isSecondFront)
    {
        sector1 = std::vector<cv::Point>(first.begin(), first.begin() + 2);
        sector2 = std::vector<cv::Point>(second.begin(), second.begin() + 2);
    }
    else if (!isFirstFront && !isSecondFront)
    {
        sector1 = std::vector<cv::Point>(first.end() - 2, first.end());
        sector2 = std::vector<cv::Point>(second.end() - 2, second.end());
    }
    return std::max(cv::arcLength(sector1, false), cv::arcLength(sector2, false));

}

void JoinSegmentsRecursive
(
    std::vector<std::vector<cv::Point>>& input, 
    std::vector<std::vector<cv::Point>>& output, 
    double toleranceLen,
    bool isFirstRun,
    ImageType imgType,
    cv::Point imgDim
)
{
    std::vector<std::vector<cv::Point>> inputVector = input;
    std::vector<std::vector<cv::Point>> jointVector;
    std::vector<std::vector<cv::Point>> outputVector;
    
    bool isVertical;
    // first sort for easier comparison
    if (isFirstRun)
    {
        for (auto& segment : inputVector)
        {
            int verticalDiff = FindMinMaxPoint(segment, true, false).y - FindMinMaxPoint(segment, false, false).y;
            int horizontalDiff = FindMinMaxPoint(segment, true, true).x - FindMinMaxPoint(segment, false, true).x;
            if (horizontalDiff < verticalDiff)
            {
                isVertical = true;
            }

            std::sort(segment.begin(), segment.end(), [isVertical](const cv::Point& a, const cv::Point& b) {
                return (isVertical ? a.y < b.y : a.x < b.x);
            });
        }
    }

    std::vector<int> used (inputVector.size(), false);
    bool anyMerged = false;
    for (int i = 0; i < inputVector.size(); i++)
    {
        if (used[i])
        {
            continue;
        }

        bool toJoin = false;
        bool duplicate = false;

        float vectorLen_i = cv::arcLength(inputVector[i], false);
                        
        for (int j = 0; j < inputVector.size(); j++)
        {
            float vectorLen_j = cv::arcLength(inputVector[j], false);
            float len = std::max(vectorLen_i, vectorLen_j);
            // float pointsTolerance = toleranceLen * len;
            if (j == i)
            {
                continue;
            }
            if (used[j])
            {
                continue;
            }
            bool closeBeginToEnd = distance(inputVector[j].front(), inputVector[i].back()) 
                            <  CalcMaxLenSector(inputVector[i], inputVector[j], false, true) * toleranceLen;
            bool closeEndToBegin = distance(inputVector[j].back(), inputVector[i].front())
                            <  CalcMaxLenSector(inputVector[i], inputVector[j], true, false) * toleranceLen;

            bool closeBeginToBegin = distance(inputVector[j].front(),inputVector[i].front())
                            < CalcMaxLenSector(inputVector[i], inputVector[j], true, true) * toleranceLen;
            bool closeEndToEnd = distance(inputVector[j].back(),inputVector[i].back())
                            < CalcMaxLenSector(inputVector[i], inputVector[j], false, false) * toleranceLen;


            bool sameBeginToBegin = distance(inputVector[j].front(),inputVector[i].front())
                                        <  len*0.01;
            bool sameEndToEnd = distance(inputVector[j].back(),inputVector[i].back())
                                        <  len*0.01;
            
             std::vector<cv::Point> joined;

            if (closeBeginToEnd || closeEndToBegin || closeBeginToBegin || closeEndToEnd)
            {
                // JOIN WHERE THEY INTERSECT!!!!! NOT DIRECTLY!!!

                std::vector<cv::Point> sector1, sector2;
                if (closeBeginToEnd)
                {
                    sector1 = std::vector<cv::Point>(inputVector[j].begin(), inputVector[j].begin() + 2);
                    sector2 = std::vector<cv::Point>(inputVector[i].end() - 2, inputVector[i].end());
                }                
                else if (closeEndToBegin)
                {
                    sector1 = std::vector<cv::Point>(inputVector[j].end() - 2, inputVector[j].end());
                    sector2 = std::vector<cv::Point>(inputVector[i].begin(), inputVector[i].begin() + 2);
                }
                else if (closeBeginToBegin)
                {
                    sector1 = std::vector<cv::Point>(inputVector[j].begin(), inputVector[j].begin() + 2);
                    sector2 = std::vector<cv::Point>(inputVector[i].begin(), inputVector[i].begin() + 2);
                }
                else if (closeEndToEnd)
                {
                    sector1 = std::vector<cv::Point>(inputVector[j].end() - 2, inputVector[j].end());
                    sector2 = std::vector<cv::Point>(inputVector[i].end() - 2, inputVector[i].end());
                }

                cv::Point2f p(sector1[0]), p2(sector1[1]);
                cv::Point2f q(sector2[0]), q2(sector2[1]);

                cv::Point2f r = p2 - p;
                cv::Point2f s = q2 - q;

                double rxs = cross2D(r, s);
                cv::Point intersection;
                if (std::abs(rxs) > 1e-9) 
                {
                    double t = cross2D(q - p, s) / rxs;
                    intersection = p + r * t;
                }
                else
                {
                    //qDebug() << "intersection far, but shouldn't be!";
                    continue;
                }
                
                if (intersection.x < 0 || intersection.y < 0 || intersection.x > imgDim.x || intersection.y > imgDim.y)
                {
                    continue;
                }
                float intersectionTolerance = 
                            std::max(cv::arcLength(sector1, false), cv::arcLength(sector2, false)) * 2;

                if (closeBeginToEnd)
                {
                    if (distance(inputVector[i].back(), intersection) < intersectionTolerance)
                    {
                        joined = inputVector[i];
                        joined.pop_back();
                        joined.push_back(intersection);
                        joined.insert(joined.end(), inputVector[j].begin() + 1, inputVector[j].end());
                    }
                }
                else if (closeEndToBegin)
                {
                    if (distance(inputVector[j].back(), intersection) < intersectionTolerance)
                    {
                        joined = inputVector[j];
                        joined.pop_back();
                        joined.push_back(intersection);
                        joined.insert(joined.end(), inputVector[i].begin() + 1, inputVector[i].end());
                    }
                }
                else if (closeBeginToBegin)
                {
                    if (distance(inputVector[i].front(), intersection) < intersectionTolerance)
                    {
                        std::vector<cv::Point> inputIReverse = inputVector[i];
                        std::reverse(inputIReverse.begin(), inputIReverse.end());
                        joined = inputIReverse;
                        joined.pop_back();
                        joined.push_back(intersection);
                        joined.insert(joined.end(), inputVector[j].begin() + 1, inputVector[j].end());
                    }
                }
                else if (closeEndToEnd)
                {
                    if (distance(inputVector[j].back(), intersection) < intersectionTolerance)
                    {
                        std::vector<cv::Point> inputJReverse = inputVector[j];
                        std::reverse(inputJReverse.begin(), inputJReverse.end());
                        joined = inputVector[i];
                        joined.pop_back();
                        joined.push_back(intersection);
                        joined.insert(joined.end(), inputJReverse.begin() + 1, inputJReverse.end());
                    }
                } 
            }

            if (joined.size() > 1)
            {
                toJoin = true;
                anyMerged = true;
                jointVector.push_back(joined);
                used[i] = true;
                used[j] = true;
                break;
            }
            
        }
        if (!toJoin)
        {
            jointVector.push_back(inputVector[i]);
            used[i] = true;
        }
    }

    if (anyMerged)
    {
        JoinSegmentsRecursive(jointVector, outputVector, toleranceLen, false);
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


void FindApexes(const std::vector<std::vector<cv::Point>>& inputVector, std::vector<cv::Point>& apexes)
{
    // cycle through pairs of vectors, check 4 distances: 
    // * between begins
    // * between ends
    // * between begin1 and end2
    // * between begin2 and end1
    // if any is < tolerance then calculate angle and find point of intersection

    std::vector<int> used (inputVector.size(), false);

    std::vector<cv::Point> apexesRaw;

    for (int i = 0; i < inputVector.size(); i++)
    {
        if (used[i])
        {
            continue;
        }

        float vectorLen = sqrt( pow((inputVector[i].front().x - inputVector[i].back().x), 2) +
                                        pow((inputVector[i].front().y - inputVector[i].back().y), 2)
                                    );
        float pointsTolerance = 0.25*vectorLen; // px

        for (int j = 0; j < inputVector.size(); j++)
        {
            if (j == i)
            {
                continue;
            }
            if (used[j])
            {
                continue;
            }

            bool distanceBeginToEnd = distance(inputVector[j].front(), inputVector[i].back()) 
                                        <  pointsTolerance;
            bool distanceEndToBegin = distance(inputVector[j].back(), inputVector[i].front())
                                        <  pointsTolerance;

            bool distanceBeginToBegin = distance(inputVector[j].front(),inputVector[i].front())
                                        <  pointsTolerance;
            bool distanceEndToEnd = distance(inputVector[j].back(),inputVector[i].back())
                                        <  pointsTolerance;

            if (distanceBeginToEnd || distanceEndToBegin || distanceBeginToBegin || distanceEndToEnd)
            {
                cv::Point2f p(inputVector[i][0]), p2(inputVector[i][1]);
                cv::Point2f q(inputVector[j][0]), q2(inputVector[j][1]);

                cv::Point2f r = p2 - p;
                cv::Point2f s = q2 - q;

                double nr = norm(r);
                double ns = norm(s);

                // Angle between directions
                double angle = 0;
                if (nr > 0 && ns > 0) 
                {
                    double cosang = dot(r, s) / (nr * ns);
                    cosang = std::max(-1.0, std::min(1.0, cosang));
                    angle = std::acos(cosang) * (180 / 3.1416);
                }

                if (angle > 5.0 && angle < 45.0)
                {
                    double rxs = cross2D(r, s);
                    if (std::abs(rxs) > 1e-9) 
                    {
                        double t = cross2D(q - p, s) / rxs;
                        cv::Point intersection = p + r * t;
                        
                        bool d1p0 = distance(cv::Point(inputVector[i][0]), intersection) < pointsTolerance;
                        bool d1p1 = distance(cv::Point(inputVector[i][1]), intersection) < pointsTolerance;

                        bool d2p0 = distance(cv::Point(inputVector[j][0]), intersection) < pointsTolerance;
                        bool d2p1 = distance(cv::Point(inputVector[j][1]), intersection) < pointsTolerance;

                        if (d1p0 || d1p1 || d2p0 || d2p1)
                        {
                            apexesRaw.push_back(intersection);
                            used[i] = true;
                            used[j] = true;
                        }
                    }
                }
            }
        }
    }

    apexes = apexesRaw;
}

ApexSanityResult FindApexes
(
    const std::vector<std::vector<cv::Point>> & shapes,
    std::vector<std::vector<cv::Point>>& electrodes,
    std::vector<cv::Point>& apexes,
    int imgWidth,
    int imgHeight
)
{
    std::vector<cv::Point> apexesVect;
    ApexSanityResult apexError = ApexSanityResult::APEX_OK;

    for (auto electrode : shapes)
    { 
        if (electrode.size() <= 2)
        {
            // too small -> not connected to anything
            continue;
        }
        // Approximate polyline with straight segments
        std::vector<cv::Point> approx;
        double epsilon = 10.0; // Approximation accuracy
        cv::approxPolyDP(electrode, approx, epsilon, false);

        // Find angles between consecutive segments
        double limitAngle = 45;
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
            if (interior_deg < limitAngle && interior_deg > 10.0)
            {  
                apex = approx[i];

                ApexSanityResult isApexOk = ApexSanityCheck(approx, i, imgWidth, imgHeight);

                if (isApexOk == ApexSanityResult::APEX_OK)
                {
                    apexesVect.push_back(apex);
                    electrodes.push_back(electrode);
                }
            }
        }
    }
    apexes = apexesVect;
    return apexError;
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
    if (this->imageType == ImageType::XRAY)
    {
        cv::GaussianBlur(imgGray, imgBlur, cv::Size(3,3), 1.7);
    }
    else
    {
        cv::bilateralFilter(imgGray, imgBlur, 7, 75, 75);       
    }

    // TRACING
    cv::Mat toTrace = img.clone();

    int lenThr = 40;
    double distThr = 1.4142;
    int canny1Thr = 10;
    int canny2Thr = 40;

    if (imageType == ImageType::PHOTO)
    {
        lenThr = 40;
        canny1Thr = 150;
        canny2Thr = 200;
    }
    cv::Ptr<cv::ximgproc::FastLineDetector> fld =
            cv::ximgproc::createFastLineDetector(lenThr,
                                                distThr,
                                                canny1Thr, canny2Thr,
                                                /*aperture=*/3, /*do_merge=*/true);

    std::vector<cv::Vec4f> lines;
    fld->detect(imgGray, lines);

    if (lines.size() < 1)
    {
        success = false;
    }
    {
        std::vector<std::vector<cv::Point>> segments;
        std::vector<std::vector<cv::Point>> electrodesJoined;
        
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

        //join broken contours
        this->lamp.electrodes.clear();
        double tolerance = 0.2; //imageType == ImageType::XRAY ? 0.2 : 0.1;
        JoinSegmentsRecursive(segments, electrodesJoined, tolerance, true, imageType, cv::Point(imgGray.cols, imgGray.rows));

        if (electrodesJoined.size() < 1)
        {
            success = false;
        }
        else
        {
            // I HAVE NO IDEA WHY, BUT DON'T TOUCH
            std::sort(electrodesJoined.begin(), electrodesJoined.end(),
                [](auto &a, auto &b)
                { 
                    return a.size() < b.size(); 
                }
            );
            while (electrodesJoined.back().size() < 3)
            {
                electrodesJoined.pop_back();
            }

            cv::Mat toShow = img.clone();
            int i = 1;
            for (auto poly : electrodesJoined)
            {
                cv::polylines(toShow, poly, 0, colours[3],3);
                cv::putText(toShow, 
                    cv::String(std::to_string(i)),  
                    poly[0],
                    cv::FONT_HERSHEY_PLAIN,
                    1.5,
                    colours[3]
                );
                i++;
            }
            cv::imshow("JOINED", toShow);

            // find vertices
            
            std::vector<cv::Point> apexesVect;
            ApexSanityResult apexError = FindApexes
                                        (
                                            electrodesJoined,
                                            this->lamp.electrodes,
                                            apexesVect,
                                            imgGray.cols,
                                            imgGray.rows
                                        );
                                                
            // might be duplicate points because contours might be duplicated
            if (apexesVect.size() > 2)
            {
                RemoveDuplicatePoints(apexesVect, apexesVect);
            }

            // only leave those that are aligned (stricly horizontal) or misalighned on a small angle
            if (apexesVect.size() > 2)
            {
                double minDistance = 1000;
                QPair<int, int> bestApexes = {0,0};
                std::vector<cv::Point> twoApexes;
                for (int i = 0; i < apexesVect.size(); ++i)
                {
                    for (int j = 1; j < apexesVect.size(); ++j)
                    {
                        if (j == i)
                        {
                            continue;
                        }
                        auto [distance, angle] = CalculateDistanceAngle(apexesVect[i], apexesVect[j]);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            bestApexes = {i, j};
                        }
                    }
                }
                if (bestApexes.first != bestApexes.second)
                {
                    twoApexes.push_back(apexesVect[bestApexes.first]);
                    twoApexes.push_back(apexesVect[bestApexes.second]);
                    apexesVect = twoApexes;
                }
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
                if (std::abs(angle) > 45)
                {
                    angle = 90 - std::abs(angle);
                }
                if (angle > 30)
                {
                    emit ElectrodesAnalysisFailed();
                    success = false;
                }
                this->lamp.lampCenter = cv::Point2f((apexes.first.x + apexes.second.x)/2, (apexes.first.y + apexes.second.y)/2);

                this->lamp.electrodesDistancePx = distance;
                this->lamp.electrodesApexes = apexes;
                this->lamp.electrodesAngle = angle;
            }

            // WRITE DOWN VERTICALITY
            std::vector<cv::Point> electrode0 = this->lamp.electrodes[0];
            cv::Point apex;
            int verticalDiff, horizontalDiff;
            
            auto it0 = std::find(electrode0.begin(), electrode0.end(), this->lamp.electrodesApexes.first);
            auto it1 = std::find(electrode0.begin(), electrode0.end(), this->lamp.electrodesApexes.second);
            if (it0 == electrode0.end())
            {
                it0 = std::find(electrode0.begin(), electrode0.end(), this->lamp.electrodesApexes.second);
            }

            if (it0 != electrode0.end())
            {
                verticalDiff = std::max(std::abs((*it0).y - (*(it0 - 1)).y), std::abs((*it0).y - (*(it0 + 1)).y));
                horizontalDiff = std::max(std::abs((*it0).x - (*(it0 - 1)).x), std::abs((*it0).x - (*(it0 + 1)).x));
            }
            else
            {
                verticalDiff = FindMinMaxPoint(electrode0, true, false).y - FindMinMaxPoint(electrode0, false, false).y;
                horizontalDiff = FindMinMaxPoint(electrode0, true, true).x - FindMinMaxPoint(electrode0, false, true).x;
            }

            this->lamp.isVertical = (horizontalDiff < verticalDiff);

            cv::Mat drawn = this->currentImageCv->clone();

            cv::drawMarker(
                drawn,
                this->lamp.electrodesApexes.first,
                cv::Scalar(0,0,0),
                cv::MARKER_CROSS,   // <- use a valid marker type
                10,                 // markerSize
                1                   // thickness
            );

            cv::drawMarker(
                drawn,
                this->lamp.electrodesApexes.second,
                cv::Scalar(0,0,0),
                cv::MARKER_CROSS,   // <- use a valid marker type
                10,                 // markerSize
                1                   // thickness
            );

            cv::imshow("apexes", drawn);
        }
    }

    return success;
}

QPair<double, double>  LampImageAnalyser::CalculateWallThickness()
{
    QPair<double, double> result = QPair<size_t, size_t>(0, 0);

    if (!(this->lamp.wallsApexes.empty()))
    {
        std::vector<cv::Point> edgesLeftTop, edgesRightBottom;
        for (cv::Point apex : this->lamp.wallsApexes)
        {
            double distanceToCenter = this->lamp.isVertical ? std::abs(this->lamp.lampCenter.y - apex.y)
                                                            : std::abs(this->lamp.lampCenter.x - apex.x);
            double centerApproxTolerance = this->lamp.electrodesDistancePx * 0.5;
            if (distanceToCenter > centerApproxTolerance)
            {
                continue;
            }
            bool leftOrTop;
            if (this->lamp.isVertical)
            {
                leftOrTop = apex.x < this->lamp.lampCenter.x;
            }
            else
            {
                leftOrTop = apex.y < this->lamp.lampCenter.y;               
            }

            if (leftOrTop && edgesLeftTop.size() < 2)
            {
                edgesLeftTop.push_back(apex);
            }
            else if (!leftOrTop && edgesRightBottom.size() < 2)
            {
                edgesRightBottom.push_back(apex);
            }
        }

        if (edgesLeftTop.size() == 2 && edgesRightBottom.size() == 2)
        {
            auto [wallLeftPx, angleLeft] = CalculateDistanceAngle(edgesLeftTop[0], edgesLeftTop[1]);
            auto [wallRightPx, angleRight] = CalculateDistanceAngle(edgesRightBottom[0], edgesRightBottom[1]);

            double ratio = this->lamp.pixelMmRatio;

            double wallLeftMm = ratio != 0 ? wallLeftPx / this->lamp.pixelMmRatio : 0;
            double wallRightMm = ratio != 0 ? wallRightPx / this->lamp.pixelMmRatio : 0;
            result.first = wallLeftMm;
            result.second = wallRightMm;
        }
    }
    return result;
}

void LampImageAnalyser::FindScale(QImage& inputImage)
{
    bool success = false;

    uint8_t * buffer = (uint8_t*)inputImage.bits();
    size_t width = inputImage.width();
    size_t height = inputImage.height();
    size_t stride = inputImage.bytesPerLine();

    cv::Mat img = cv::Mat(height, width, CV_8UC3, buffer, stride);

    cv::Mat imgRoi = img(cv::Rect(0, 0, width/4, height/4));

    cv::Mat imgGray;
    cv::cvtColor(imgRoi, imgGray, cv::COLOR_BGR2GRAY);
    
    cv::Mat cannied;
    cv::Canny(imgGray, cannied, 10, 200, 3);
    
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(cannied, lines, 1, CV_PI/180, 50, 50, 10 );

    double bestLen = 0.0;
    int bestIdx = -1;
    if (lines.size() > 0)
    {
        // cv::Mat lined = img.clone();
        // for( size_t i = 0; i < lines.size(); i++ )
        // {
        //     cv::Vec4i l = lines[i];
        //     cv::line( lined, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA);
        // }
        // cv::imshow("lined", lined);

        for (int i = 0; i < lines.size(); ++i)
        {
            const double dx = double(lines[i][2]) - double(lines[i][0]);
            const double dy = double(lines[i][3]) - double(lines[i][1]);
            double angle = std::abs(std::atan2(dy, dx));

            double maxRad = 5 * CV_PI/ 180.0;
            
            if ((angle < maxRad) || (std::abs(CV_PI - angle) < maxRad))
            {
                double L = std::sqrt(dx*dx + dy*dy);
                if (L > bestLen)
                {
                    bestLen = L;
                    bestIdx = i;
                }
            }
        }
        if (bestIdx > 0)
        {
            this->lamp.pixelMmRatio = bestLen;
            success = true;
        }
    }

    if (!success)
    {
        this->lamp.pixelMmRatio = pxToMmDefaultPhoto; // very very rough estimate
    }
}


void LampImageAnalyser::RunFullAnalysis(QImage& imgToProcess)
{
    bool success = true;
    this->lamp.ResetLampModel();

    this->fullResultsString = "";

    if (this->imageType == ImageType::PHOTO)
    {
        FindScale(imgToProcess);
    }
    else
    {
        this->lamp.pixelMmRatio = pxToMmDefaultXray;
    }

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

            emit ElectrodesProcessed(coordElectrodeFirst, 
                                    coordElectrodeSecond, 
                                    electrodesDistanceMm,
                                    electrodesAngle
                                    );
            emit LampWallsProcessed(this->lamp.wallsCenters, 
                                    this->lamp.wallsRadii, 
                                    this->lamp.wallsThickness, 
                                    this->lamp.lampWidth
                                    );

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

void LampImageAnalyser::SetImageType(ImageType type)
{
    this->imageType = type;
}
