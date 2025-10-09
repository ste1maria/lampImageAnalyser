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
#include "doubleCurve.hpp"

cv::Mat imgBlur;
int thresh = 5;
std::vector<std::vector<cv::Point> > contours;

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
};

void LampImageAnalyser::SetCurrentImage(const QImage& image)
{
    cv::Mat view(image.height(), image.width(), CV_8UC3,
                 const_cast<uchar*>(image.bits()), image.bytesPerLine());
    *this->currentImageCv = view.clone(); 
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

void LampImageAnalyser::RunEdgeDetection(QImage& inputImage)
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
        

        cv::Mat coloredEdges; 
        cv::cvtColor(imgGray, coloredEdges, cv::COLOR_GRAY2BGR);

        for (size_t i = 0; i < this->contours.size(); ++i) 
        {
            cv::polylines(coloredEdges, this->contours[i], false, colours[i], 1, cv::LINE_AA);
        }
        //cv::Mat edgedImageBW;
        //cv::cvtColor(coloredEdges, edgedImageBW, cv::COLOR_BGR2GRAY);
        
        this->edgeDetectedImage = contouredImg;
        *this->currentImageCv = coloredEdges;

        emit ImageProcessed(this->currentImageCv);

        CalculateRadii();
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

void LampImageAnalyser::PaintCircleCenter(cv::Point point, int curveNumber, cv::Scalar col)
{
    cv::circle(*this->currentImageCv, point, 3, col, cv::FILLED);
    cv::String coordinates = cv::String(std::format("({},{})", point.x, point.y));

    cv::putText(*this->currentImageCv, 
                coordinates, 
                cv::Point(point.x + 5, point.y + 5),
                cv::FONT_HERSHEY_PLAIN,
                1,
               col
    );
    emit ShowVerticeOnImage(this->currentImageCv);
}

// 1) resample the polyline by arc length (open/closed)
static std::vector<cv::Point2f> resampleByArcLen(const std::vector<cv::Point2f>& P, float ds, bool closed=false) {
    std::vector<cv::Point2f> Q; 
    Q.reserve(P.size()+1);
    for (auto& p : P) 
        Q.emplace_back(p);
    if (closed && (Q.front()!=Q.back())) 
        Q.push_back(Q.front());

    std::vector<float> S(Q.size(), 0.f);
    for (size_t i=1;i<Q.size();++i) 
        S[i] = S[i-1] + cv::norm(Q[i]-Q[i-1]);
    float L = S.back(); 
    if (L <= ds) 
        return Q;

    int M = std::max(2, int(std::floor(L/ds))+1);
    std::vector<cv::Point2f> R; 
    R.reserve(M);
    for (int m=0; m<M; ++m) 
    {
        float s = std::min(m*ds, L);
        auto it = std::lower_bound(S.begin(), S.end(), s);
        size_t j = std::clamp<size_t>(std::max<int>(1, int(it - S.begin())), 1, Q.size()-1);
        float t = (S[j]>S[j-1]) ? (s - S[j-1])/(S[j]-S[j-1]) : 0.f;
        R.emplace_back(Q[j-1] + t*(Q[j]-Q[j-1]));
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
    if (poly.size() < 2) return;
    auto toInt = [](const cv::Point2f& p){ return cv::Point(cvRound(p.x), cvRound(p.y)); };
    for (size_t i = 1; i < poly.size(); ++i)
        cv::line(img, toInt(poly[i-1]), toInt(poly[i]), color, thickness, lineType);
    if (closed)
        cv::line(img, toInt(poly.back()), toInt(poly.front()), color, thickness, lineType);
}

struct RadiusStats {
    float r_min;       // minimum local radius (tightest bend)
    float r_p05;       // 5th percentile radius (robust tightest bend)
    float r_median;    // median local radius
    float r_mean;      // mean of local radii
    float r_hmean;     // harmonic mean of local radii (averaging curvature)
    float r_fit;       // global best-fit circle radius (Kåsa)
    cv::Point2f c_fit; // best-fit circle center
    int n_samples;     // points used for local stats
};


static bool circumcircleR(const cv::Point2f& A, const cv::Point2f& B, const cv::Point2f& C,
                          cv::Point2f& center, float& radius)
{
    float a2=A.dot(A), b2=B.dot(B), c2=C.dot(C);
    float d = 2.f*(A.x*(B.y-C.y) + B.x*(C.y-A.y) + C.x*(A.y-B.y));
    if (std::abs(d) < 1e-8f) return false; // nearly colinear
    center.x = (a2*(B.y-C.y) + b2*(C.y-A.y) + c2*(A.y-B.y)) / d;
    center.y = (a2*(C.x-B.x) + b2*(A.x-C.x) + c2*(B.x-A.x)) / d;
    radius   = cv::norm(center - B);
    return true;
}

static bool fitCircleKasa(const std::vector<cv::Point2f>& P, cv::Point2f& C, float& R) {
    if (P.size() < 3) return false;
    cv::Mat A(P.size(), 3, CV_64F), b(P.size(), 1, CV_64F);
    for (int i=0;i<(int)P.size();++i) {
        double x=P[i].x, y=P[i].y;
        A.at<double>(i,0)=x; A.at<double>(i,1)=y; A.at<double>(i,2)=1.0;
        b.at<double>(i,0)=-(x*x + y*y);
    }
    cv::Mat sol;
    if (!cv::solve(A, b, sol, cv::DECOMP_SVD)) return false;
    double Acoef=sol.at<double>(0), Bcoef=sol.at<double>(1), Ccoef=sol.at<double>(2);
    double cx = -Acoef/2.0, cy = -Bcoef/2.0;
    double r2 = cx*cx + cy*cy - Ccoef;
    if (r2 <= 0) return false;
    C = cv::Point2f((float)cx,(float)cy);
    R = (float)std::sqrt(r2);
    return std::isfinite(R);
}

template<typename T>
static T percentile(std::vector<T> v, double p01_to_99) {
    if (v.empty()) return std::numeric_limits<T>::quiet_NaN();
    p01_to_99 = std::clamp(p01_to_99, 0.0, 100.0);
    size_t k = (size_t)std::floor((p01_to_99/100.0) * (v.size()-1));
    std::nth_element(v.begin(), v.begin()+k, v.end());
    return v[k];
}

// local radii via 3-point circumcircle on a uniformly-sampled curve
static std::vector<float> localRadii(const std::vector<cv::Point2f>& P, float ds=1.0f, int stride=2, bool closed=false) {
    auto U = resampleByArcLen(P, ds, closed);
    int n = (int)U.size();
    if (n < 3) return {};
    auto idx = [&](int i){ return closed ? ( (i%n+n)%n ) : std::clamp(i,0,n-1); };
    std::vector<float> R; R.reserve(n);
    for (int i=0;i<n;++i) {
        int i0 = idx(i - stride), i2 = idx(i + stride);
        cv::Point2f c; float r;
        if (circumcircleR(U[i0], U[i], U[i2], c, r) && std::isfinite(r)) R.push_back(r);
    }
    return R;
}

// ---------- main: stats per polyline ----------
static RadiusStats curvatureRadiusForPolyline(const std::vector<cv::Point2f>& poly,
                                              float ds=1.0f, int stride=2, bool closedFlag=false)
{
    bool closed = false;
    auto Rv = localRadii(poly, ds, stride, closed);
    // keep finite and cap absurdly huge radii for stats robustness
    std::vector<float> Rs;
    Rs.reserve(Rv.size());
    for (float r : Rv) if (std::isfinite(r)) Rs.push_back(std::min(r, 1e6f));

    RadiusStats S{};
    S.n_samples = (int)Rs.size();
    if (!Rs.empty()) {
        S.r_min    = *std::min_element(Rs.begin(), Rs.end());
        S.r_p05    = percentile(Rs, 5.0);
        S.r_median = percentile(Rs, 50.0);
        S.r_mean   = std::accumulate(Rs.begin(), Rs.end(), 0.0f)/Rs.size();

        // harmonic mean on curvature (average kappa = mean(1/r)), then 1/mean_kappa
        double sumInv = 0.0;
        int cntInv = 0;
        for (float r : Rs) { if (r > 1e-9f){ sumInv += 1.0/r; ++cntInv; } }
        S.r_hmean = (cntInv>0) ? float(cntInv / sumInv) : std::numeric_limits<float>::infinity();
    } else {
        S.r_min=S.r_p05=S.r_median=S.r_mean=S.r_hmean=std::numeric_limits<float>::quiet_NaN();
    }

    // global circle fit (good if the curve is an arc)
    fitCircleKasa(poly, S.c_fit, S.r_fit);
    return S;
}

// batch version: one stats record per polyline
static std::vector<RadiusStats> curvatureRadiusForPolylines(const std::vector<std::vector<cv::Point2f>>& polys,
                                                            float ds=1.0f, int stride=2, bool closed=false)
{
    std::vector<RadiusStats> out; out.reserve(polys.size());
    for (auto& p : polys) out.push_back(curvatureRadiusForPolyline(p, ds, stride, closed));
    return out;
}


typedef enum 
{
    RIGHT,
    LEFT,
    AMBIG
} Orientation;

// to find the apex
int findApex(std::vector<cv::Point> &curve, Orientation orientation)
{
    int max_x = 0;
    int min_x = 10000;
    int apexIndex = 0;
    if (orientation == Orientation::AMBIG)
    {
        return 0;
    }
    for (size_t i = 0; i < curve.size(); i++)
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
    return apexIndex;
}

void LampImageAnalyser::CalculateRadii()
{
    bool closed = false;        // set true for closed shapes

    //cv::Mat temp = this->currentImageCv->clone();

    std::vector<std::vector<cv::Point2f>> simples;
    std::vector<int>colour_indices;
    for (size_t i = 0; i < this->contours.size(); i++)
    {
        if (contours[i].size() < 200)
        {  
            continue;
        }
        std::vector<cv::Point> contour = contours[i];

        std::sort(contour.begin(), contour.end(), [](const cv::Point& a, const cv::Point& b) {
            return a.y < b.y;
        });
        
        OpeningResult r = detectOpeningLR(contour);
        Orientation orient =  (r.opening == OpeningLR::Left ? Orientation::LEFT :
                        r.opening == OpeningLR::Right ? Orientation::RIGHT : Orientation::AMBIG);

        int apexIdx = findApex(contour, orient);

        // only leave 70% from the curve centered at the apex
        int index_begin = apexIdx - apexIdx*0.35;
        int index_end = apexIdx + (contour.size() - apexIdx)*0.35;

        colour_indices.push_back(i);
        std::vector<cv::Point2f> simple(contour.begin() + index_begin, contour.begin() + index_end);
        simples.push_back(simple);
        //drawPolylineF(temp, simple, closed, colours[i+1], 2, cv::LINE_AA);
    }
    //cv::imshow("smoothed", temp);

    auto stats = curvatureRadiusForPolylines(simples, /*ds=*/1.0f, /*stride=*/3, /*closed=*/false);
    
    QString radii;

    for (size_t i=0; i<stats.size(); ++i) 
    {
        const auto& s = stats[i];
        qDebug() << "Curve " << i << ": fitR=" << s.r_fit << " @ (" << s.c_fit.x << "," << s.c_fit.y;

        std::string radius(std::format("({}) R={:.1f} px", i + 1, s.r_fit));
        radii += QString::fromStdString(radius) + "; ";
        
        cv::Point coordinates = cv::Point(contours[i][0].x - 50, contours[i][0].y + 20);

        cv::putText(*this->currentImageCv, 
                cv::String(radius), 
                coordinates,
                cv::FONT_HERSHEY_PLAIN,
                0.8,
                colours[colour_indices[i]]
        );

        PaintCircleCenter(cv::Point(s.c_fit.x, s.c_fit.y), i + 1, colours[colour_indices[i]]);
    }
    
    emit RadiiCalculated(radii);
}
