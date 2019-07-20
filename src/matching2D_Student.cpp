#include <numeric>
#include "matching2D.hpp"

using namespace std;

namespace
{
    // Detect keypoints in image using the traditional detectors
    void detKeypointsTraditional(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, bool useHarris)
    {
        std::string detName = useHarris ? "Harris" : "Shi-Tomasi";

        // compute detector parameters based on image size
        int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        double maxOverlap = 0.0; // max. permissible overlap between two features in %
        double minDistance = (1.0 - maxOverlap) * blockSize;
        int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

        double qualityLevel = useHarris ? 0.001 : 0.01; // minimal accepted quality of image corners
        double k = 0.04;

        // Apply corner detection
        double t = (double)cv::getTickCount();
        vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarris, k);

        // add corners to result vector
        for (auto it = corners.begin(); it != corners.end(); ++it)
        {
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
            newKeyPoint.size = blockSize;
            keypoints.push_back(newKeyPoint);
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << detName <<" detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

        // visualize results
        if (bVis)
        {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = detName + " Corner Detector Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }
    }
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descSource.type() == CV_32F ? cv::NORM_L2 : cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround, float 32 type is required
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        vector<vector<cv::DMatch>> knn_matches;
        
        // k nearest neighbors (k=2)
        matcher->knnMatch(descSource, descRef, knn_matches, 2);

        // distance ratio filtering
        for (int i = 0; i < kPtsSource.size(); ++i) {
            if (knn_matches[i][0].distance < 0.8 * knn_matches[i][1].distance) {
                matches.push_back(knn_matches[i][0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // SIFT
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        constexpr int threshold = 30;        // FAST/AGAST detection threshold score.
        constexpr int octaves = 3;           // detection octaves (use 0 to do single scale)
        constexpr float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        constexpr int bytes = 32;
        constexpr bool use_orientation = false;

        extractor = cv::ORB::create(bytes, use_orientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        constexpr int nfeatures = 500;
        constexpr float scaleFactor = 1.2f;
        constexpr int nlevels = 8;
        constexpr int edgeThreshold = 31;
        constexpr int firstLevel = 0;
        constexpr int WTA_K = 2;
        constexpr cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        constexpr int patchSize = 31;
        constexpr int fastThreshold = 20;

        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        constexpr bool orientationNormalized = true;
        constexpr bool scaleNormalized = true;
        constexpr float patternScale = 22.0f;
        constexpr int nOctaves = 4;

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        constexpr cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        constexpr int descriptor_size = 0;
        constexpr int descriptor_channels = 3;
        constexpr float threshold = 0.001f;
        constexpr int nOctaves = 4;
        constexpr int nOctaveLayers = 4;
        constexpr cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        constexpr int nfeatures = 0;
        constexpr int nOctaveLayers = 3;
        constexpr double contrastThreshold = 0.04;
        constexpr double edgeThreshold = 10;
        constexpr double sigma = 1.6;

        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // BRISK detector / descriptor
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("BRISK") == 0)
    {
        constexpr int threshold = 30;        // FAST/AGAST detection threshold score.
        constexpr int octaves = 3;           // detection octaves (use 0 to do single scale)
        constexpr float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        
        detector = cv::BRISK::create();
    } else if (detectorType.compare("FAST") == 0) {
        constexpr int threshold = 10;
        constexpr bool nonmaxSuppression = true;
        constexpr cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;

        detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
    } else if (detectorType.compare("ORB") == 0) {
        constexpr int nfeatures = 500;
        constexpr float scaleFactor = 1.2f;
        constexpr int nlevels = 8;
        constexpr int edgeThreshold = 31;
        constexpr int firstLevel = 0;
        constexpr int WTA_K = 2;
        constexpr cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        constexpr int patchSize = 31;
        constexpr int fastThreshold = 20;

        detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    } else if (detectorType.compare("AKAZE") == 0) {
        constexpr cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        constexpr int descriptor_size = 0;
        constexpr int descriptor_channels = 3;
        constexpr float threshold = 0.001f;
        constexpr int nOctaves = 4;
        constexpr int nOctaveLayers = 4;
        constexpr cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

        detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    } else if (detectorType.compare("SIFT") == 0) {
        constexpr int nfeatures = 0;
        constexpr int nOctaveLayers = 3;
        constexpr double contrastThreshold = 0.04;
        constexpr double edgeThreshold = 10;
        constexpr double sigma = 1.6;

        detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis/*=false*/)
{
    detKeypointsTraditional(keypoints, img, bVis, true);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis/*=false*/)
{
    detKeypointsTraditional(keypoints, img, bVis, false);
}
