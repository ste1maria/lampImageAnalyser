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

int thresh = 5;

cv::Scalar colours[] = {
    cv::Scalar(0, 0, 0), 
    cv::Scalar(255, 0, 0), 
    cv::Scalar(255, 0, 255), 
    cv::Scalar(0, 0, 255),
    cv::Scalar(10, 255, 0),
    cv::Scalar(0, 255, 255),
    cv::Scalar(64, 64, 255),
    cv::Scalar(255, 128, 128)
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
}

cv::Mat pruneSpurs(const cv::Mat& thin255, int iterations = 6) {
    CV_Assert(thin255.type() == CV_8U);
    cv::Mat skel; cv::threshold(thin255, skel, 0, 1, cv::THRESH_BINARY); // 0/1

    // 8-neighborhood count (ignore center)
    cv::Mat K = cv::Mat::ones(3,3,CV_8U); K.at<uchar>(1,1) = 0;

    for (int i = 0; i < iterations; ++i) {
        cv::Mat nbrs;
        cv::filter2D(skel, nbrs, CV_8U, K, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
        cv::Mat endpoints = (skel == 1) & (nbrs == 1);  // degree==1
        if (cv::countNonZero(endpoints) == 0) break;
        skel.setTo(0, endpoints);
    }
    skel *= 255;
    return skel;
}

void LampImageAnalyser::ProcessLampWalls(QImage& inputImage)
{
    uint8_t * buffer = (uint8_t*)inputImage.bits();
    size_t width = inputImage.width();
    size_t height = inputImage.height();
    size_t stride = inputImage.bytesPerLine();

    try
    {
        cv::Mat img = cv::Mat(height, width, CV_8UC3, buffer, stride);
        cv::Mat imgGray;
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
        cv::Mat imgBlur;

        cv::blur(imgGray, imgBlur, cv::Size(10,10));
        // SOBEL
        cv::Mat sobelx, sobely, gradient;
        // Apply Sobel operator
        cv::Sobel(imgBlur, sobelx, CV_64F, 1, 0, 3);
        cv::Sobel(imgBlur, sobely, CV_64F, 0, 1, 3);
    
        // Compute gradient magnitude
        cv::magnitude(sobelx, sobely, gradient);

        // Convert to 8-bit image
        cv::Mat gradient_abs;
        cv::convertScaleAbs(gradient, gradient_abs);        

        cv::Mat edges;
        cv::threshold(gradient_abs, edges, 10, 255, cv::THRESH_BINARY);
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
        //cv::imshow("SOBEL", edges);

        cv::Mat labels, stats, centroids;
        int n = cv::connectedComponentsWithStats(edges, labels, stats, centroids, 8, CV_32S);
        int min_area = 50; // tune
        for (int i = 1; i < n; ++i) {
            if (stats.at<int>(i, cv::CC_STAT_AREA) < min_area)
            {
                edges.setTo(0, labels == i);
            }
        }
        //cv::imshow("SOBEL shaved", edges);

        // THINNING
        cv::Mat thinned_edges;
        cv::ximgproc::thinning(edges, thinned_edges, cv::ximgproc::THINNING_ZHANGSUEN);
        //cv::imshow("thinned", thinned_edges);

        cv::Mat contouredImg = pruneSpurs(thinned_edges, 10);

        // TRACING
        cv::Mat toTrace = contouredImg.clone();
        this->contours.clear();
        cv::findContours(toTrace, this->contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        std::sort(this->contours.begin(), this->contours.end(),
              [](auto &a, auto &b){ return a.size() > b.size(); });
        
        if (this->contours.size() > 10 || this->contours.size() < 4)
        {
            emit WallsAnalysisFailed();
            return;
        }

        if (this->lamp.lampCenter == this->lamp.undeterminedPoint)
        {
            this->CalculateElectrodes(inputImage);
        }
        this->SimplifyLampContours();

        this->CalculateRadii();

        emit ImageProcessed(this->currentImageCv);
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
    cv::circle(*this->currentImageCv, point, 3, col, cv::FILLED);
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

struct RadiusStats {
    float r_min;       // minimum local radius (tightest bend)
    float r_p05;       // 5th percentile radius (robust tightest bend)
    float r_median;    // median local radius
    float r_mean;      // mean of local radii
    float r_hmean;     // harmonic mean of local radii (averaging curvature)
    float r_fit;       // global best-fit circle radius (Kåsa)
    cv::Point c_fit; // best-fit circle center
    int n_samples;     // points used for local stats
};


static bool circumcircleR(const cv::Point& A, const cv::Point& B, const cv::Point& C,
                          cv::Point& center, float& radius)
{
    float a2 = A.dot(A), b2 = B.dot(B), c2 = C.dot(C);
    float d = 2.f*(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));

    if (std::abs(d) < 1e-8f)
    {
        return false; // nearly colinear
    }
    center.x = (a2 * (B.y - C.y) + b2 * (C.y - A.y) + c2 * (A.y - B.y)) / d;
    center.y = (a2 * (C.x - B.x) + b2 * (A.x - C.x) + c2 * (B.x - A.x)) / d;
    radius   = cv::norm(center - B);
    return true;
}

static bool fitCircleKasa(const std::vector<cv::Point>& P, cv::Point& C, float& R) 
{
    if (P.size() < 3) 
    {
        return false;
    }
    cv::Mat A(P.size(), 3, CV_64F), b(P.size(), 1, CV_64F);

    for (int i = 0; i < (int)P.size(); ++i)
    {
        double x = P[i].x, y = P[i].y;
        A.at<double>(i, 0) = x; 
        A.at<double>(i, 1) = y; 
        A.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) =- (x*x + y*y);
    }
    cv::Mat sol;
    if (!cv::solve(A, b, sol, cv::DECOMP_SVD)) 
    {
        return false;
    }
    double Acoef = sol.at<double>(0), Bcoef = sol.at<double>(1), Ccoef = sol.at<double>(2);
    double cx = -Acoef/2.0, cy = -Bcoef/2.0;
    double r2 = cx*cx + cy*cy - Ccoef;
    if (r2 <= 0) 
    {
        return false;
    }
    C = cv::Point((float)cx, (float)cy);
    R = (float)std::sqrt(r2);
    return std::isfinite(R);
}

template<typename T>
static T percentile(std::vector<T> v, double p01_to_99) 
{
    if (v.empty()) 
    {
        return std::numeric_limits<T>::quiet_NaN();
    }
    p01_to_99 = std::clamp(p01_to_99, 0.0, 100.0);
    size_t k = (size_t)std::floor((p01_to_99/100.0) * (v.size()-1));
    std::nth_element(v.begin(), v.begin()+k, v.end());
    return v[k];
}

// local radii via 3-point circumcircle on a uniformly-sampled curve
static std::vector<float> localRadii(const std::vector<cv::Point>& P, float ds=1.0f, int stride=2, bool closed=false) 
{
    auto U = resampleByArcLen(P, ds, closed);
    int n = (int)U.size();

    if (n < 3)
    {
        return {};
    }
    
    auto idx = [&](int i)
    { 
        return closed ? ( (i % n + n) % n ) : std::clamp(i, 0, n-1); 
    };

    std::vector<float> R; 
    R.reserve(n);

    for (int i = 0; i < n; ++i) 
    {
        int i0 = idx(i - stride), i2 = idx(i + stride);
        cv::Point c; 
        float r;
        if (circumcircleR(U[i0], U[i], U[i2], c, r) && std::isfinite(r)) 
        {
            R.push_back(r);
        }
    }
    return R;
}

// ---------- main: stats per polyline ----------
static RadiusStats curvatureRadiusForPolyline(const std::vector<cv::Point>& poly,
                                              float ds=1.0f, int stride=2, bool closedFlag=false)
{
    bool closed = false;
    auto Rv = localRadii(poly, ds, stride, closed);
    // keep finite and cap absurdly huge radii for stats robustness
    std::vector<float> Rs;
    Rs.reserve(Rv.size());
    for (float r : Rv) 
    {
        if (std::isfinite(r)) 
        {
            Rs.push_back(std::min(r, 1e6f));
        }
    }

    RadiusStats S{};
    S.n_samples = (int)Rs.size();
    if (!Rs.empty()) 
    {
        S.r_min    = *std::min_element(Rs.begin(), Rs.end());
        S.r_p05    = percentile(Rs, 5.0);
        S.r_median = percentile(Rs, 50.0);
        S.r_mean   = std::accumulate(Rs.begin(), Rs.end(), 0.0f)/Rs.size();

        // harmonic mean on curvature (average kappa = mean(1/r)), then 1/mean_kappa
        double sumInv = 0.0;
        int cntInv = 0;
        for (float r : Rs) 
        { 
            if (r > 1e-9f){ sumInv += 1.0/r; ++cntInv; } 
        }
        S.r_hmean = (cntInv>0) ? float(cntInv / sumInv) : std::numeric_limits<float>::infinity();
    } 
    else 
    {
        S.r_min=S.r_p05=S.r_median=S.r_mean=S.r_hmean=std::numeric_limits<float>::quiet_NaN();
    }

    // global circle fit (good if the curve is an arc)
    fitCircleKasa(poly, S.c_fit, S.r_fit);
    return S;
}

// batch version: one stats record per polyline
static std::vector<RadiusStats> curvatureRadiusForPolylines(const std::vector<std::vector<cv::Point>>& polys,
                                                            float ds=1.0f, int stride=2, bool closed=false)
{
    std::vector<RadiusStats> out; 
    out.reserve(polys.size());
    for (auto& p : polys)
    {
        out.push_back(curvatureRadiusForPolyline(p, ds, stride, closed));
    }
        
    return out;
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
    int minDiff = 10;

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

void LampImageAnalyser::FindPixelMmRatio(int lampWidthPx)
{
    this->lamp.pixelMmRatio = lampWidthPx / standardLampWidth_mm;
}

void LampImageAnalyser::SimplifyLampContours()
{
    std::vector<std::vector<cv::Point>> simples;
    std::vector<std::vector<cv::Point>> trueContours;
    int leftmostApexX(0), rightmostApexX(0);

    // find curve apexes
    for (size_t i = 0; i < this->contours.size(); i++)
    {
        // throwing away contours that do not resemble lamp walls
        if (this->contours[i].size() < 200)
        {  
            continue;
        }
        std::vector<cv::Point> contour = this->contours[i];

        // sorting because some segments are repeating
        std::sort(contour.begin(), contour.end(), [](const cv::Point& a, const cv::Point& b) {
            return a.y < b.y;
        });
        
        OpeningResult r = detectOpeningLR(contour);
        Orientation orient =  (r.opening == OpeningLR::Left ? Orientation::LEFT :
                                r.opening == OpeningLR::Right ? Orientation::RIGHT : Orientation::AMBIG);

        int apexIdx = findCurveApex(contour, orient, this->lamp.lampCenter.y);
        
        this->lamp.lampWallsApexes.push_back(contour[apexIdx]);

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
    }

    // find pixel to mm ratio from lamp width
    this->FindPixelMmRatio(rightmostApexX - leftmostApexX);
    int pixels2mm = 2 * this->lamp.pixelMmRatio;

    for (size_t i = 0; i < trueContours.size(); i++)
    {
        unsigned bestIdx_up(0), bestIdx_down(0);
        unsigned bestAbs_up(10), bestAbs_down(10);// maximum tolerance to Y difference

        cv::Point currentApex = this->lamp.lampWallsApexes[i];

        int y_up = currentApex.y - pixels2mm;
        int y_down = currentApex.y + pixels2mm;
        for (size_t j = 0; j < trueContours[i].size(); j++)
        {
            if (j < trueContours[i].size() / 2)
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
        std::vector<cv::Point> simple(trueContours[i].begin() + bestIdx_up, trueContours[i].begin() + bestIdx_down);
        simples.push_back(simple);
    }

    this->lamp.lampContours = simples;

    cv::Mat drawn = this->currentImageCv->clone();
    for (size_t i = 0; i < this->lamp.lampContours.size(); ++i) 
    {
        cv::polylines(drawn, this->lamp.lampContours[i], false, colours[i], 1, cv::LINE_AA);
    }
    *this->currentImageCv = std::move(drawn);
}

void LampImageAnalyser::CalculateRadii()
{
    auto stats = curvatureRadiusForPolylines(this->lamp.lampContours, /*ds=*/1.0f, /*stride=*/3, /*closed=*/false);
    
    QString radii;

    cv::Mat drawn = this->currentImageCv->clone();
    for (size_t i=0; i<stats.size(); ++i) 
    {
        const auto& s = stats[i];
        qDebug() << "Curve " << i << ": fitR=" << s.r_fit << " @ (" << s.c_fit.x << "," << s.c_fit.y;

        std::string radius(std::format("({}) R={:.1f} px", i + 1, s.r_fit));
        std::string curveNumber(std::format("({})", i + 1));
        radii += QString::fromStdString(radius) + "; ";
        
        cv::Point coordinates = cv::Point(this->lamp.lampContours[i][0].x, this->lamp.lampContours[i][0].y - 30);

        cv::putText(drawn, 
                cv::String(curveNumber), 
                coordinates,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                colours[i]
        );

        PaintCircleCenter(cv::Point(s.c_fit.x, s.c_fit.y), i + 1, colours[i]);
    }
    
    *this->currentImageCv = std::move(drawn);
    emit RadiiCalculated(radii);
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

void LampImageAnalyser::ProcessElectrodes(QImage& inputImage)
{
    try
    {
        bool success = this->CalculateElectrodes(inputImage);
        if (success)
        {
            PaintVertice(this->lamp.electrodesApexes.first);
            PaintVertice(this->lamp.electrodesApexes.second);
            QPoint coordFirst(this->lamp.electrodesApexes.first.x, this->lamp.electrodesApexes.first.y);
            QPoint coordSecond(this->lamp.electrodesApexes.second.x, this->lamp.electrodesApexes.second.y);
            size_t distance = this->lamp.electrodesDistance;
            emit ElectrodesProcessed(coordFirst, coordSecond, distance);
            // cv::circle(coloredEdges, this->lamp.lampCenter, 3, cv::Scalar(0, 0, 255));
            // //qDebug() << "distance" << distance << "angle" << angle;
            // qDebug() << "center" << this->lamp.lampCenter.x << this->lamp.lampCenter.y;
        }
    }
    catch(...)
    {
        qDebug() << "Exception while processing lamp electrodes";
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
    cv::GaussianBlur(imgGray, imgBlur, cv::Size(3,3), 1.2);

    cv::Mat cannyImg;
    cv::Canny(imgBlur, cannyImg, 0, 100);

    // TRACING
    cv::Mat toTrace = cannyImg.clone();
    this->lamp.electrodes.clear();
    cv::findContours(toTrace, this->lamp.electrodes, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::sort(this->lamp.electrodes.begin(), this->lamp.electrodes.end(),
            [](auto &a, auto &b){ return a.size() > b.size(); });

    // find vertices
    std::vector<cv::Point> apexesVect;
    ApexSanityResult apexError = ApexSanityResult::APEX_OK;
    for (auto electrode : this->lamp.electrodes) 
    {
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
        std::vector<cv::Point> approx_sorted = approx;
        // sometimes the lamp.contours have double segments or go back and forth
        std::sort(approx_sorted.begin(), approx_sorted.end(), [](const cv::Point& a, const cv::Point& b) {
            return a.x < b.x;
        });
        ApexSanityResult isApexOk = ApexSanityCheck(approx_sorted, apex, imgGray.rows);

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
        this->lamp.electrodesDistance = static_cast<size_t>(distance);
        this->lamp.electrodesApexes = apexes;
    }
    return success;
}