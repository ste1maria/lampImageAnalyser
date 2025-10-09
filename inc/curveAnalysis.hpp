// curve_orientation.hpp
#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>

enum class OpeningLR { Left = -1, Ambiguous = 0, Right = 1 };

struct OpeningResult {
    OpeningLR opening = OpeningLR::Ambiguous;
    float confidence = 0.f;   // 0..1 heuristic
    bool fromQuadratic = false;
    float a = 0.f;            // quadratic coefficient in x' = a y'^2 + b y' + c
};

// --- PCA rotate: make Y' the long axis ---
inline void rotateToPrincipalY(const std::vector<cv::Point>& pts,
                               cv::Mat& rot /*Nx2 float*/,
                               cv::Point& mean,
                               cv::Mat& eig /*2x2 float, rows=eigenvectors*/)
{
    CV_Assert(pts.size() >= 2);
    cv::Mat data((int)pts.size(), 2, CV_32F);
    for (int i = 0; i < (int)pts.size(); ++i) {
        data.at<float>(i,0) = pts[i].x;
        data.at<float>(i,1) = pts[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    mean = { pca.mean.at<float>(0,0), pca.mean.at<float>(0,1) };
    eig = pca.eigenvectors.clone(); // row0 major, row1 minor
    cv::Mat R = eig.t(); // columns are eigenvectors

    rot.create((int)pts.size(), 2, CV_32F);
    for (int i = 0; i < (int)pts.size(); ++i) {
        cv::Vec2f p(pts[i].x - mean.x, pts[i].y - mean.y);
        cv::Vec2f q(
            R.at<float>(0,0)*p[0] + R.at<float>(0,1)*p[1],
            R.at<float>(1,0)*p[0] + R.at<float>(1,1)*p[1]
        );
        rot.at<float>(i,0) = q[0];
        rot.at<float>(i,1) = q[1];
    }

    // Ensure Y' is the long axis: if spreadX > spreadY, swap axes
    cv::Scalar meanR, stdR;
    cv::meanStdDev(rot, meanR, stdR);
    if (stdR[0] > stdR[1]) {
        for (int i = 0; i < rot.rows; ++i)
            std::swap(rot.at<float>(i,0), rot.at<float>(i,1));
        // Swap eig rows so inverse stays consistent with our local axes choice
        cv::Mat es = eig.clone();
        eig.at<float>(0,0) = es.at<float>(1,0);
        eig.at<float>(0,1) = es.at<float>(1,1);
        eig.at<float>(1,0) = es.at<float>(0,0);
        eig.at<float>(1,1) = es.at<float>(0,1);
    }
}

// Quadratic fit: x' = a y'^2 + b y' + c
inline bool fitQuadraticXY(const cv::Mat& rot, float& a, float& b, float& c, float* r2_out = nullptr)
{
    const int N = rot.rows;
    if (N < 5) return false;

    double Sy=0, Sy2=0, Sy3=0, Sy4=0, Sx=0, Sxy=0, Sy2x=0;
    double SS_tot = 0.0, SS_res = 0.0;

    // Precompute mean x' for R^2
    double meanX = 0.0;
    for (int i=0;i<N;++i) meanX += rot.at<float>(i,0);
    meanX /= N;

    for (int i = 0; i < N; ++i) {
        double y = rot.at<float>(i,1);
        double x = rot.at<float>(i,0);
        double y2 = y*y;
        Sy   += y;
        Sy2  += y2;
        Sy3  += y2*y;
        Sy4  += y2*y2;
        Sx   += x;
        Sxy  += x*y;
        Sy2x += x*y2;
        double dx = x - meanX;
        SS_tot += dx*dx;
    }

    cv::Mat1d A(3,3), rhs(3,1);
    A << Sy4, Sy3, Sy2,
         Sy3, Sy2, Sy,
         Sy2, Sy,  (double)N;
    rhs << Sy2x, Sxy, Sx;

    cv::Mat1d sol;
    if (!cv::solve(A, rhs, sol, cv::DECOMP_SVD)) return false;
    a = (float)sol(0); b = (float)sol(1); c = (float)sol(2);

    if (std::abs(a) < 1e-7f) { if (r2_out) *r2_out = 0.f; return false; }

    // Compute R^2 to assess quality
    for (int i=0;i<N;++i) {
        double y = rot.at<float>(i,1);
        double x = rot.at<float>(i,0);
        double xp = a*y*y + b*y + c;
        double r = x - xp;
        SS_res += r*r;
    }
    if (r2_out) *r2_out = SS_tot > 1e-12 ? (float)(1.0 - SS_res/SS_tot) : 0.f;
    return true;
}

// Fallback: sign of discrete second derivative of x'(y') after sorting by y'
inline float secondDerivativeSignFallback(cv::Mat rot)
{
    // sort rows by y'
    std::vector<int> idx(rot.rows);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j){
        return rot.at<float>(i,1) < rot.at<float>(j,1);
    });

    // finite second difference of x' with respect to y' (non-uniform)
    double acc = 0.0;
    int count = 0;
    for (int k = 1; k+1 < (int)idx.size(); ++k) {
        int i0 = idx[k-1], i1 = idx[k], i2 = idx[k+1];
        double y0 = rot.at<float>(i0,1), y1 = rot.at<float>(i1,1), y2 = rot.at<float>(i2,1);
        double x0 = rot.at<float>(i0,0), x1 = rot.at<float>(i1,0), x2 = rot.at<float>(i2,0);
        double dy0 = y1 - y0, dy1 = y2 - y1;
        if (std::abs(dy0) < 1e-9 || std::abs(dy1) < 1e-9) continue;

        // piecewise second derivative estimate (for non-uniform spacing):
        // x'' ≈ 2 * ( (x2 - x1)/dy1 - (x1 - x0)/dy0 ) / (dy0 + dy1)
        double sdd = 2.0 * ( (x2 - x1)/dy1 - (x1 - x0)/dy0 ) / (dy0 + dy1);
        acc += sdd;
        ++count;
    }
    if (count == 0) return 0.f;
    double meanSdd = acc / count;
    // squash to [-1,1] for confidence hint
    double mag = std::min(1.0, std::abs(meanSdd) * 1.0); // scale can be tuned
    return (float)((meanSdd >= 0 ? 1.0 : -1.0) * mag);
}

// --- Main API: detect left/right opening ---
inline OpeningResult detectOpeningLR(const std::vector<cv::Point>& curve)
{
    OpeningResult res;
    const int N = (int)curve.size();
    if (N < 3) { res.opening = OpeningLR::Ambiguous; res.confidence = 0.f; return res; }

    cv::Mat rot; cv::Point mean; cv::Mat eig;
    rotateToPrincipalY(curve, rot, mean, eig);

    // Fit quadratic x'(y')
    float a=0.f, b=0.f, c=0.f, r2=0.f;
    bool ok = fitQuadraticXY(rot, a, b, c, &r2);
    if (ok) {
        // Heuristic confidence from |a| (stronger curvature) and R^2 (fit quality)
        float curvStrength = std::min(1.f, std::abs(a) *  (float)std::pow(cv::norm(rot, cv::NORM_L2)/N, 0.0)); // keep simple
        float conf = 0.6f * std::clamp(r2, 0.f, 1.f) + 0.4f * std::min(1.f, std::abs(a) * 100.0f); // scale 100 is empirical
        res.fromQuadratic = true;
        res.a = a;
        if (a > 0) { res.opening = OpeningLR::Right; res.confidence = conf; }
        else       { res.opening = OpeningLR::Left;  res.confidence = conf; }
        // If confidence is still weak, we can blend with fallback sign:
        if (res.confidence < 0.4f) {
            float s = secondDerivativeSignFallback(rot);
            if (s != 0.f) {
                // If fallback disagrees strongly, downgrade to Ambiguous
                if ((s > 0 && res.opening == OpeningLR::Left) ||
                    (s < 0 && res.opening == OpeningLR::Right)) {
                    res.opening = OpeningLR::Ambiguous;
                    res.confidence = 0.25f;
                }
            }
        }
        return res;
    }

    // Fallback: discrete second derivative sign
    float s = secondDerivativeSignFallback(rot); // [-1..1], sign gives orientation
    if (s > 0.f) { res.opening = OpeningLR::Right; res.confidence = std::min(1.f, s); }
    else if (s < 0.f) { res.opening = OpeningLR::Left; res.confidence = std::min(1.f, -s); }
    else { res.opening = OpeningLR::Ambiguous; res.confidence = 0.f; }
    res.fromQuadratic = false;
    res.a = 0.f;
    return res;
}
