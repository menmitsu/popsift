/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <popsift/common/device_prop.h>
#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/sift_config.h>
#include <popsift/version.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/features2d.hpp"

#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"

#ifdef USE_DEVIL
#include <devil_cpp_wrapper.hpp>
#endif
#include "pgmread.h"

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

using namespace std;

static bool print_dev_info  {false};
static bool print_time_info {false};
static bool write_as_uchar  {false};
static bool dont_write      {false};
static bool pgmread_loading {false};

static void parseargs(int argc, char** argv, popsift::Config& config, string& lFile, string& rFile) {
    using namespace boost::program_options;

    options_description options("Options");
    {
        options.add_options()
            ("help,h", "Print usage")
            ("verbose,v", bool_switch()->notifier([&](bool i) {if(i) config.setVerbose(); }), "")
            ("log", bool_switch()->notifier([&](bool i) {if(i) config.setLogMode(popsift::Config::All); }), "Write debugging files")

            ("left,l",  value<std::string>(&lFile)->required(), "\"Left\"  input file")
            ("right,r", value<std::string>(&rFile)->required(), "\"Right\" input file");

    }
    options_description parameters("Parameters");
    {
        parameters.add_options()
            ("octaves", value<int>(&config.octaves), "Number of octaves")
            ("levels", value<int>(&config.levels), "Number of levels per octave")
            ("sigma", value<float>()->notifier([&](float f) { config.setSigma(f); }), "Initial sigma value")

            ("threshold", value<float>()->notifier([&](float f) { config.setThreshold(f); }), "Contrast threshold")
            ("edge-threshold", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
            ("edge-limit", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
            ("downsampling", value<float>()->notifier([&](float f) { config.setDownsampling(f); }), "Downscale width and height of input by 2^N")
            ("initial-blur", value<float>()->notifier([&](float f) {config.setInitialBlur(f); }), "Assume initial blur, subtract when blurring first time");
    }
    options_description modes("Modes");
    {
    modes.add_options()
        ( "gauss-mode", value<std::string>()->notifier([&](const std::string& s) { config.setGaussMode(s); }),
          popsift::Config::getGaussModeUsage() )
        ("desc-mode", value<std::string>()->notifier([&](const std::string& s) { config.setDescMode(s); }),
        "Choice of descriptor extraction modes:\n"
        "loop, iloop, grid, igrid, notile\n"
	"Default is loop\n"
        "loop is OpenCV-like horizontal scanning, computing only valid points, grid extracts only useful points but rounds them, iloop uses linear texture and rotated gradiant fetching. igrid is grid with linear interpolation. notile is like igrid but avoids redundant gradiant fetching.")
        ("popsift-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::PopSift); }),
        "During the initial upscale, shift pixels by 1. In extrema refinement, steps up to 0.6, do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh. Shift feature coords octave 0 back to original pos.")
        ("vlfeat-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::VLFeat); }),
        "During the initial upscale, shift pixels by 1. That creates a sharper upscaled image. "
        "In extrema refinement, steps up to 0.6, levels remain unchanged, "
        "do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh.")
        ("opencv-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::OpenCV); }),
        "During the initial upscale, shift pixels by 0.5. "
        "In extrema refinement, steps up to 0.5, "
        "reject points when reaching max iterations, "
        "first contrast threshold is floor(.5 * peak thresh). "
        "Computed filter width are lower than VLFeat/PopSift")
        ("direct-scaling", bool_switch()->notifier([&](bool b) { if(b) config.setScalingMode(popsift::Config::ScaleDirect); }),
         "Direct each octave from upscaled orig instead of blurred level.")
        ("norm-multi", value<int>()->notifier([&](int i) {config.setNormalizationMultiplier(i); }), "Multiply the descriptor by pow(2,<int>).")
        ( "norm-mode", value<std::string>()->notifier([&](const std::string& s) { config.setNormMode(s); }),
          popsift::Config::getNormModeUsage() )
        ( "root-sift", bool_switch()->notifier([&](bool b) { if(b) config.setNormMode(popsift::Config::RootSift); }),
          popsift::Config::getNormModeUsage() )
        ("filter-max-extrema", value<int>()->notifier([&](int f) {config.setFilterMaxExtrema(f); }), "Approximate max number of extrema.")
        ("filter-grid", value<int>()->notifier([&](int f) {config.setFilterGridSize(f); }), "Grid edge length for extrema filtering (ie. value 4 leads to a 4x4 grid)")
        ("filter-sort", value<std::string>()->notifier([&](const std::string& s) {config.setFilterSorting(s); }), "Sort extrema in each cell by scale, either random (default), up or down");

    }
    options_description informational("Informational");
    {
        informational.add_options()
        ("print-gauss-tables", bool_switch()->notifier([&](bool b) { if(b) config.setPrintGaussTables(); }), "A debug output printing Gauss filter size and tables")
        ("print-dev-info", bool_switch(&print_dev_info)->default_value(false), "A debug output printing CUDA device information")
        ("print-time-info", bool_switch(&print_time_info)->default_value(false), "A debug output printing image processing time after load()")
        ("write-as-uchar", bool_switch(&write_as_uchar)->default_value(false), "Output descriptors rounded to int Scaling to sensible ranges is not automatic, should be combined with --norm-multi=9 or similar")
        ("dont-write", bool_switch(&dont_write)->default_value(false), "Suppress descriptor output")
        ("pgmread-loading", bool_switch(&pgmread_loading)->default_value(false), "Use the old image loader instead of LibDevIL")
        ;

        //("test-direct-scaling")
    }

    options_description all("Allowed options");
    all.add(options).add(parameters).add(modes).add(informational);
    variables_map vm;

    try
    {
       store(parse_command_line(argc, argv, all), vm);

       if (vm.count("help")) {
           std::cout << all << '\n';
           exit(1);
       }

        notify(vm); // Notify does processing (e.g., raise exceptions if required args are missing)
    }
    catch(boost::program_options::error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl;
        std::cerr << "Usage:\n\n" << all << std::endl;
        exit(EXIT_FAILURE);
    }
}


static void collectFilenames( list<string>& inputFiles, const boost::filesystem::path& inputFile )
{
    vector<boost::filesystem::path> vec;
    std::copy( boost::filesystem::directory_iterator( inputFile ),
               boost::filesystem::directory_iterator(),
               std::back_inserter(vec) );
    for (const auto& currPath : vec)
    {
        if( boost::filesystem::is_regular_file(currPath) )
        {
            inputFiles.push_back( currPath.string() );

        }
        else if( boost::filesystem::is_directory(currPath) )
        {
            collectFilenames( inputFiles, currPath);
        }
    }
}

SiftJob* process_image( const string& inputFile, PopSift& PopSift )
{
    unsigned char* image_data;
    SiftJob* job;

    nvtxRangePushA( "load and convert image" );
#ifdef USE_DEVIL
    if( ! pgmread_loading )
    {
        ilImage img;
        if( img.Load( inputFile.c_str() ) == false ) {
            cerr << "Could not load image " << inputFile << endl;
            return 0;
        }
        if( img.Convert( IL_LUMINANCE ) == false ) {
            cerr << "Failed converting image " << inputFile << " to unsigned greyscale image" << endl;
            exit( -1 );
        }
        const auto w = img.Width();
        const auto h = img.Height();
        cout << "Loading " << w << " x " << h << " image " << inputFile << endl;
        image_data = img.GetData();

        nvtxRangePop( );

        // PopSift.init( w, h );
        job = PopSift.enqueue( w, h, image_data );

        img.Clear();
    }
    else
#endif
    {
        int h{};
        int w{};
        image_data = readPGMfile( inputFile, w, h );
        if( image_data == nullptr ) {
            exit( EXIT_FAILURE );
        }

        nvtxRangePop( );

        // PopSift.init( w, h );
        job = PopSift.enqueue( w, h, image_data );

        delete [] image_data;
    }

    return job;
}

cv::Mat convertFeaturesDevToMat(popsift::FeaturesDev* lFeatures, popsift::FeaturesHost* left_features)
{

  left_features->pin( );
  cudaMemcpy( left_features->getFeatures(),
                      lFeatures->getFeatures(),
                      lFeatures->getFeatureCount() * sizeof(popsift::Feature),
                      cudaMemcpyDeviceToHost );
  cudaMemcpy( left_features->getDescriptors(),
                      lFeatures->getDescriptors(),
                      lFeatures->getDescriptorCount() * sizeof(popsift::Descriptor),
                      cudaMemcpyDeviceToHost );
  left_features->unpin( );

  std::vector<popsift::Feature> featuresVec(left_features->getFeatures(), left_features->getFeatures() +  lFeatures->getFeatureCount());
  std::vector<popsift::Descriptor> descriptors(left_features->getDescriptors(), left_features->getDescriptors() +  lFeatures->getDescriptorCount());

  std::vector<const float*> descriptors_fl;

  cout<<featuresVec[0].xpos<<"  "<<featuresVec[0].ypos;
  cout<<"\n"<<left_features->getDescriptors();

  float* lptr  = (float*)( left_features->getDescriptors());
  // cv::Mat mat1;
   cv::Mat mat1(lFeatures->getDescriptorCount(),128,CV_32FC1, lptr);


}

int main(int argc, char **argv)
{
    cudaDeviceReset();

    cout << "OpenCV version : " << CV_VERSION << endl;

  cout << "Major version : " << CV_MAJOR_VERSION << endl;
  cout << "Minor version : " << CV_MINOR_VERSION << endl;

  cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;




    popsift::Config config;
    string         lFile{};
    string         rFile{};

    std::cout << "PopSift version: " << POPSIFT_VERSION_STRING << std::endl;

    try {
        parseargs( argc, argv, config, lFile, rFile ); // Parse command line
        std::cout << lFile << " <-> " << rFile << std::endl;
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return EXIT_SUCCESS;
    }

    if( boost::filesystem::exists( lFile ) ) {
        if( ! boost::filesystem::is_regular_file( lFile ) ) {
            cout << "Input file " << lFile << " is not a regular file, nothing to do" << endl;
            return EXIT_FAILURE;
        }
    }

    if( boost::filesystem::exists( rFile ) ) {
        if( ! boost::filesystem::is_regular_file( rFile ) ) {
            cout << "Input file " << rFile << " is not a regular file, nothing to do" << endl;
            return EXIT_FAILURE;
        }
    }

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set( 0, print_dev_info );
    if( print_dev_info ) deviceInfo.print( );

    PopSift PopSift( config, popsift::Config::MatchingMode );

    SiftJob* lJob = process_image( lFile, PopSift );
    SiftJob* rJob = process_image( rFile, PopSift );

    popsift::FeaturesDev* lFeatures = lJob->getDev();

    cout << "Number of features:    " << lFeatures->getFeatureCount() << endl;
    cout << "Number of descriptors: " << lFeatures->getDescriptorCount() << endl;

    popsift::FeaturesDev* rFeatures = rJob->getDev();
    cout << "Number of features:    " << rFeatures->getFeatureCount() << endl;
    cout << "Number of descriptors: " << rFeatures->getDescriptorCount() << endl;

    // lFeatures->match( rFeatures );

    popsift::FeaturesHost* left_features = new popsift::FeaturesHost( lFeatures->getFeatureCount(), lFeatures->getDescriptorCount() );
    popsift::FeaturesHost* right_features = new popsift::FeaturesHost( rFeatures->getFeatureCount(), rFeatures->getDescriptorCount() );


    // cv::Mat leftDescMat=convertFeaturesDevToMat(lFeatures,left_features);
    // cout<<leftDescMat;


    std::vector<const float*> descriptors_fl;

    cv::Mat limg=cv::imread(lFile);
    cv::Mat outLimg=limg.clone();

    cv::Mat rimg=cv::imread(rFile);
    cv::Mat outRimg=rimg.clone();


/////

    left_features->pin( );
    cudaMemcpy( left_features->getFeatures(),
                        lFeatures->getFeatures(),
                        lFeatures->getFeatureCount() * sizeof(popsift::Feature),
                        cudaMemcpyDeviceToHost );
    cudaMemcpy( left_features->getDescriptors(),
                        lFeatures->getDescriptors(),
                        lFeatures->getDescriptorCount() * sizeof(popsift::Descriptor),
                        cudaMemcpyDeviceToHost );
    left_features->unpin( );

    right_features->pin( );
    cudaMemcpy( right_features->getFeatures(),
                        rFeatures->getFeatures(),
                        rFeatures->getFeatureCount() * sizeof(popsift::Feature),
                        cudaMemcpyDeviceToHost );
    cudaMemcpy( right_features->getDescriptors(),
                        rFeatures->getDescriptors(),
                        rFeatures->getDescriptorCount() * sizeof(popsift::Descriptor),
                        cudaMemcpyDeviceToHost );
    right_features->unpin( );



    std::vector<popsift::Feature> leftFeaturesVec(left_features->getFeatures(), left_features->getFeatures() +  lFeatures->getFeatureCount());
    std::vector<popsift::Descriptor> leftFeaturesDesc(left_features->getDescriptors(), left_features->getDescriptors() +  lFeatures->getDescriptorCount());

    std::vector<cv::KeyPoint> leftKpts,rightKpts;

    std::vector<popsift::Feature> rightFeaturesVec(right_features->getFeatures(), right_features->getFeatures() +  rFeatures->getFeatureCount());
    std::vector<popsift::Descriptor> rightFeaturesDesc(right_features->getDescriptors(), right_features->getDescriptors() +  rFeatures->getDescriptorCount());


    for(int i=0;i<leftFeaturesVec.size();i++)
    {
      leftKpts.push_back(cv::KeyPoint((int)leftFeaturesVec[i].xpos,(int)leftFeaturesVec[i].ypos,-1));
    }

    for(int i=0;i<rightFeaturesVec.size();i++)
    {
      rightKpts.push_back(cv::KeyPoint((int)rightFeaturesVec[i].xpos,(int)rightFeaturesVec[i].ypos,-1));
    }

    cv::drawKeypoints(limg,leftKpts,outLimg);

    cv::imshow("Left Keypoints",outLimg);

    cv::drawKeypoints(rimg,rightKpts,outRimg);
    cv::imshow("Right Keypoints",outRimg);

    // std::ofstream of( "output-features-l.txt" );
    // lFeatures->print( of, write_as_uchar );

    cout<<leftFeaturesVec[0].xpos<<"  "<<leftFeaturesVec[0].ypos<<"\n";

    float* descPtr  = (float*)( leftFeaturesVec[0].desc);
    cv::Mat descMatPtr(1,128,CV_8UC1, descPtr);


     cout<<descMatPtr<<"\n\n";


    // cout<<"\n"<<left_features->getDescriptors();

    float* lptr  = (float*)( left_features->getDescriptors());

    cv::Mat lDescMat(lFeatures->getDescriptorCount(),128,CV_32FC1, lptr);
    cout<<lDescMat.row(1)<<"\n\n\n";

    cv::FileStorage file("lDescMat.mat", cv::FileStorage::WRITE);
    file<<"LMat"<<lDescMat;


        delete lFeatures;
        delete rFeatures;

        PopSift.uninit( );



    cout<<"\n\n\n\n";


    float* rptr  = (float*)(right_features->getDescriptors());
    cv::Mat rDescMat(rFeatures->getDescriptorCount(),128,CV_32FC1, rptr);
    cout<<rDescMat.row(0)<<"\n\n\n";




    cv::Ptr<cv::DescriptorMatcher> matcher =cv:: DescriptorMatcher::create("BruteForce");

    std::vector< std::vector<cv::DMatch> > cpuKnnMatches;
    std::vector<cv::DMatch> good_matches;

     matcher->knnMatch( lDescMat, rDescMat, cpuKnnMatches,2);

     // const float ratio_thresh = 0.7f;

    double max_dist = 0;
      double min_dist = 100;
//
//   for(int i=0;i<cpuKnnMatches.size();i++)
//   {
//      double dist = cpuKnnMatches[i].distance;
//
//
//      // std::cout<<"\n"<<dist<<"\t"<<cpuKnnMatches[i].queryIdx<<"  "<<cpuKnnMatches[i].trainIdx;
//      if (dist < min_dist)
//           min_dist = dist;
//      if (dist > max_dist)
//           max_dist = dist;
//
// }


cv::Mat matchesImg;
//
// for(int i=0;i<cpuKnnMatches.size();i++)
// {
//    if( cpuKnnMatches[i].distance <= max(2*min_dist, 0.02) )
//     {
//       cout<<"\n\n"<<leftFeaturesVec[cpuKnnMatches[i].queryIdx].xpos<<"    "<<leftFeaturesVec[cpuKnnMatches[i].queryIdx].ypos<<"   "<<rightFeaturesVec[cpuKnnMatches[i].trainIdx].xpos<<"   "<<rightFeaturesVec[cpuKnnMatches[i].trainIdx].ypos;
//       cout<<"\n\n"<<limg.cols*leftFeaturesVec[cpuKnnMatches[i].queryIdx].xpos<<"    "<<limg.rows*leftFeaturesVec[cpuKnnMatches[i].queryIdx].ypos<<"   "<<rimg.cols*rightFeaturesVec[cpuKnnMatches[i].trainIdx].xpos<<"   "<<rimg.rows*rightFeaturesVec[cpuKnnMatches[i].trainIdx].ypos;
//       good_matches.push_back( cpuKnnMatches[i]);
//     }
// }


    const float ratio_thresh = 0.7f;

    for (size_t i = 0; i < cpuKnnMatches.size(); i++)
    {
        if (cpuKnnMatches[i][0].distance < ratio_thresh * cpuKnnMatches[i][1].distance)
        {
            good_matches.push_back(cpuKnnMatches[i][0]);

            // cout<<"\n Good match found at "<<leftKpts[good_matches[i].queryIdx].pt<<" \t\t"<<rightKpts[good_matches[i].trainIdx].pt;

        }
    }


// cv::waitKey(0);

    // drawMatches( outLimg, leftKpts, outRimg, rightKpts, good_matches, matchesImg, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    // cv::imshow("Matches",matchesImg);

cout<<"\n\n"<<"Total Matches"<<cpuKnnMatches.size()<<" out of which good matches are:  "<<good_matches.size()<<"\n";

cout<<"Distances"<<min_dist<<"    \t\t\t"<<max_dist;




//
// for(int i=0;i<leftFeaturesVec.size();i++)
// {
//   cv::circle(limg, cv::Point(leftFeaturesVec[i].xpos,leftFeaturesVec[i].ypos),4, cv::Scalar(0,0,255));
// }
//
//
//
// for(int i=0;i<rightFeaturesVec.size();i++)
// {
//   cv::circle(rimg, cv::Point(rightFeaturesVec[i].xpos,rightFeaturesVec[i].ypos),4, cv::Scalar(0,0,255));
// }
//







// matchesImg=cv::Mat::ones(std::max(limg.rows,rimg.rows),limg.cols+rimg.cols,CV_8UC3);

std::cout<<limg.channels()<<"   "<<rimg.channels()<<"  "<<matchesImg.channels();

std::cout<<"Rows"<<limg.rows<<"   "<<rimg.rows<<"  "<<matchesImg.rows;
std::cout<<"Cols"<<limg.cols<<"   "<<rimg.cols<<"  "<<matchesImg.cols;


std::cout<<"\n Matches image"<<matchesImg.rows<< " "<<matchesImg.cols<<std::endl;
//
// limg.copyTo(matchesImg(cv::Rect(0,0,limg.cols,limg.rows)));
// rimg.copyTo(matchesImg(cv::Rect(limg.cols,0,rimg.cols,rimg.rows)));

//
for(int i=0;i<good_matches.size();i++)
{

  line(matchesImg, cv::Point(leftKpts[good_matches[i].queryIdx].pt.x,leftKpts[good_matches[i].queryIdx].pt.y),  cv::Point(rightKpts[good_matches[i].trainIdx].pt.x+limg.cols,rightKpts[good_matches[i].trainIdx].pt.y), cv::Scalar(255, 0, 0),
     4, cv::LINE_8);


}


cv::imshow("Left img",limg);
cv::imshow("Right img",rimg);
cv::waitKey(0);














//
// Ptr<DescriptorMatcher> matcher2 = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
//   std::vector< std::vector<DMatch> > knn_matches;



///////









     // cout<<mat1.rows<<mat1.cols<<endl;



    // std::vector<float> descriptorsvec(lptr,lptr+128);
    // cout<<descriptorsvec.size();
    //
    // for(int i=0;i<descriptorsvec.size();i++)
    // cout<<"\n"<<descriptorsvec[i]<<"\t";



    return EXIT_SUCCESS;
}
