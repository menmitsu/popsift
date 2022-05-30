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
#include <ctime>
#include <math.h>

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

using namespace std;

#define PI 3.14159265


using namespace cv;

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

SiftJob* detectFeaturesAndDescriptors(cv::Mat img,PopSift& PopSift)
{

  unsigned char* image_data;
  SiftJob* job;
  cv::Mat greyMat;
  cv::cvtColor(img, greyMat, cv::COLOR_BGR2GRAY);
  // imwrite("input_test.pgm",greyMat);

  nvtxRangePushA( "load and convert image" );

  // {
  //     int h{};
  //     int w{};
  //     image_data = readPGMfile( "input_test.pgm", w, h );
  //     if( image_data == nullptr ) {
  //         exit( EXIT_FAILURE );
  //     }
  //
  //     nvtxRangePop( );
  //
  //     // PopSift.init( w, h );
  //     job = PopSift.enqueue( w, h, image_data );
  //
  //     delete [] image_data;
  // }


  image_data = greyMat.data;

  nvtxRangePop( );

  job = PopSift.enqueue( img.cols, img.rows, image_data );

return job;
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
//
vector<vector<float>>  convertPopsiftStructsToOpenCV(popsift::Features* feature_list,vector <cv::KeyPoint> &keypoints, Mat &descriptors)
{

    keypoints.clear();
    // descriptors.clear();

    keypoints.reserve(feature_list->getFeatureCount());
    // descriptors.reserve(feature_list->getFeatureCount());

    float descriptors2[128*feature_list->getFeatureCount()];
    // float descriptors2=new float(128*feature_list->getFeatureCount());
    vector<vector<float>> testDesc;

    descriptors = cv::Mat(feature_list->getFeatureCount(), 128  , CV_32F);

    for(int j=0;j<feature_list->getFeatureCount();j++)
    {

      // cv::Point pt=cv::Point((int)feature_list->_ext[j].xpos,(int)feature_list->_ext[j].ypos);
      keypoints.push_back(cv::KeyPoint (feature_list->_ext[j].xpos,feature_list->_ext[j].ypos,1));

      memcpy(descriptors.row(j).data,feature_list->_ext[j].desc[0],128*sizeof(float));

    }




return testDesc;
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


    cv::Mat limg=cv::imread(lFile);
    cv::Mat outLimg=limg.clone();

    cv::Mat rimg=cv::imread(rFile);
    cv::Mat outRimg=rimg.clone();


    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set( 0, print_dev_info );
    if( print_dev_info ) deviceInfo.print( );

    PopSift PopSift( config, popsift::Config::ExtractingMode );


    string streamName="Footage/footage_jan_8.mp4";
    // string streamName="Footage/27Jan/Take2/test11_short.mp4";



     cv::VideoCapture video(streamName.c_str());
    //VideoCapture video("udpsrc port=5000 ! application/x-rtp, media=(string)video, payload=(int)96, clock-rate=(int)90000, encoding-name=(string)H264, packetization-mode=(string)1, profile-level-id=(string)640028, ssrc=(uint)3572028551, timestamp-offset=(uint)600628017, seqnum-offset=(uint)516, a-framerate=(string)15 ! queue ! rtph264depay !  decodebin ! queue !  videoconvert ! appsink",CAP_GSTREAMER);

    cv::VideoWriter outputWriter;
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    // cv::Size captureSize = cv::Size( (int) video.get(cv::CAP_PROP_FRAME_WIDTH)+limg.cols,max((int) video.get(cv::CAP_PROP_FRAME_HEIGHT),limg.rows));
    cv::Size captureSize = cv::Size( (int) video.get(cv::CAP_PROP_FRAME_WIDTH)+limg.cols,std::max((int) video.get(cv::CAP_PROP_FRAME_HEIGHT),limg.rows));

    outputWriter.open(streamName.substr(0, streamName.length()-3)+"avi", codec, 30, captureSize, true);

    // Detect Features and Descriptors in Left image

    clock_t begin_time = clock();

    SiftJob* lJob = detectFeaturesAndDescriptors(cv::imread(lFile),PopSift);
    std::cout <<"\n\nSift detection and description time"<< float( clock () - begin_time ) /  CLOCKS_PER_SEC;

    popsift::Features* lFeatures = lJob->get();

    cout << "Number of features:    " << lFeatures->getFeatureCount() << endl;
    cout << "Number of descriptors: " << lFeatures->getDescriptorCount() << endl;

    vector<cv::KeyPoint> lKeypoints,rKeypoints;
    Mat lDescriptors,rDescriptors;

    Mat lImage,rImage;

    vector<vector<float>> testLDesc=convertPopsiftStructsToOpenCV(lFeatures,lKeypoints,lDescriptors);

    std::cout <<"\n\nSift detection and description"<< float( clock () - begin_time ) /  CLOCKS_PER_SEC;

    cv::Mat colorFrame,img2;

    SiftJob* rJob = detectFeaturesAndDescriptors( cv::imread(rFile), PopSift );


    popsift::Features* rFeatures = rJob->get();

    cout << "Number of features:    " << rFeatures->getFeatureCount() << endl;
    cout << "Number of descriptors: " << rFeatures->getDescriptorCount() << endl;

    vector<vector<float>> testRDesc=convertPopsiftStructsToOpenCV(rFeatures,rKeypoints,rDescriptors);

    std::vector<cv::Point2f> scene_corners(4);
    std::vector<cv::Point2f> obj_corners(4);


    drawKeypoints(limg,lKeypoints,lImage);

    drawKeypoints(rimg,rKeypoints,rImage);

    cout<<"Left Keypoints"<<lKeypoints.size();

    cv::imshow("Left Image",lImage);
    cv::imshow("Right Image",rImage);



    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    std::vector< std::vector<DMatch> > knn_matches;

    matcher->knnMatch( lDescriptors, rDescriptors, knn_matches, 2 );

    //-- Filter matches using the Lowe's ratio test
     float ratio_thresh = 0.5f;
    std::vector<DMatch> good_matches;
    Mat img_Matches;

    double max_dist = 0; double min_dist = 100;

    for (size_t i = 0; i < knn_matches.size(); i++)
    {

      // cout<<knn_matches[i][0].distance<<"    "<<knn_matches[i][1].distance<<endl;
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    std::cout <<"\n\nSift detection and description"<< float( clock () - begin_time ) /  CLOCKS_PER_SEC;


drawMatches( limg, lKeypoints, rimg, rKeypoints, good_matches, img_Matches, Scalar::all(-1),
  Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  std::vector<Point2f> obj;
      std::vector<Point2f> scene;

if(good_matches.size()>8)
{
        for( size_t i = 0; i < good_matches.size(); i++ )
      {
          //-- Get the keypoints from the good matches
          obj.push_back( lKeypoints[ good_matches[i].queryIdx ].pt );
          scene.push_back( rKeypoints[ good_matches[i].trainIdx ].pt );
      }

        cv::Mat H = cv::findHomography( obj, scene,cv::RANSAC );


            std::cout <<"\n\n Matching and Homography "<< float( clock () - begin_time ) /  CLOCKS_PER_SEC;

            obj_corners[0] = cv::Point2f(0, 0);
            obj_corners[1] = cv::Point2f( (float)limg.cols, 0 );
            obj_corners[2] = cv::Point2f( (float)limg.cols, (float)limg.rows );
            obj_corners[3] = cv::Point2f( 0, (float)limg.rows );

            perspectiveTransform( obj_corners, scene_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            line( img_Matches, scene_corners[0] + cv::Point2f((float)limg.cols, 0),
                  scene_corners[1] + cv::Point2f((float)limg.cols, 0), cv::Scalar(0, 255, 0), 4 );
            line( img_Matches, scene_corners[1] + cv::Point2f((float)limg.cols, 0),
                  scene_corners[2] + cv::Point2f((float)limg.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            line( img_Matches, scene_corners[2] + cv::Point2f((float)limg.cols, 0),
                  scene_corners[3] + cv::Point2f((float)limg.cols, 0), cv::Scalar( 0, 255, 0), 4 );
            line( img_Matches, scene_corners[3] + cv::Point2f((float)limg.cols, 0),
                  scene_corners[0] + cv::Point2f((float)limg.cols, 0), cv::Scalar( 0, 255, 0), 4 );
}




cv::imshow("Image matches",img_Matches);


Mat newFrame;
 ratio_thresh = 0.6f;


 float cameraIntrinsicsData[]={1.25161154e+03,0.00000000e+00,5.17238336e+02,0.00000000e+00,1.26100809e+03,3.28576642e+02,0.00000000e+00,0.00000000e+00,1.00000000e+00};

 cv::Mat cameraIntrinsicsMat(3,3,CV_32F,cameraIntrinsicsData);

 vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
 double angle,angle_range_threshold=15;

while(cv::waitKey(1)!=2)
{
    begin_time = clock();

    video>>img2;
    colorFrame=img2.clone();


    if(colorFrame.rows==0)
    continue;

    imshow("Current frame",colorFrame);
    // cv::cvtColor(colorFrame,img2, cv::COLOR_BGR2GRAY);

    SiftJob* rJob = detectFeaturesAndDescriptors( colorFrame, PopSift );

    popsift::Features* rFeatures = rJob->get();

    cout << "Number of features:    " << rFeatures->getFeatureCount() << endl;
    cout << "Number of descriptors: " << rFeatures->getDescriptorCount() << endl;

    // if(rFeatures->getDescriptorCount()<50)
    // continue;
    convertPopsiftStructsToOpenCV(rFeatures,rKeypoints,rDescriptors);

    knn_matches.clear();
    matcher->knnMatch( lDescriptors, rDescriptors, knn_matches, 2 );

    //-- Filter matches using the Lowe's ratio test

    std::vector<DMatch> good_matches;
    Mat img_Matches;

    good_matches.clear();

    for (size_t i = 0; i < knn_matches.size(); i++)
    {

      // cout<<knn_matches[i][0].distance<<"    "<<knn_matches[i][1].distance<<endl;
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

//
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  float determinant;



    if(good_matches.size()>12)
    {
      drawMatches( limg, lKeypoints, colorFrame, rKeypoints, good_matches, img_Matches, Scalar::all(-1),
       Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


          for( size_t i = 0; i < good_matches.size(); i++ )
          {
              //-- Get the keypoints from the good matches
              obj.push_back( lKeypoints[ good_matches[i].queryIdx ].pt );
              scene.push_back( rKeypoints[ good_matches[i].trainIdx ].pt );
          }

          cv::Mat H = cv::findHomography( obj, scene,cv::RANSAC );

          if(!H.empty())
          {

            determinant=cv::determinant(H(Rect ( 0, 0, 1, 1 )));
            int solutions=cv::decomposeHomographyMat(H,cameraIntrinsicsMat,Rs_decomp, ts_decomp, normals_decomp);
            angle=atan2(Rs_decomp[0].at<double>(1,0), Rs_decomp[0].at<double>(0,0));
            angle*=(180/PI);
            angle=(int)((int)angle+360)%360;

            cout<<"Atta bag angle is"<<angle;



            cv::putText(img_Matches, //target image
            "Angle is"+to_string(angle), //text
            cv::Point(limg.cols,800), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            2.0,
            CV_RGB(255,255,255), //font color
            2);


            if( (angle>360-angle_range_threshold && angle<360) || angle<angle_range_threshold && angle>0)
            putText(img_Matches,"Atta packet is upright",cv::Point(limg.cols,900),cv::FONT_HERSHEY_DUPLEX, 2.0, CV_RGB(255,255,255),2);
            else if((angle>90-angle_range_threshold && angle<90+angle_range_threshold))
            putText(img_Matches,"Right edge of atta packet facing downwards",cv::Point(limg.cols,900),cv::FONT_HERSHEY_DUPLEX, 2.0, CV_RGB(255,255,255),2);
            else if((angle>180-angle_range_threshold && angle<180+angle_range_threshold))
            putText(img_Matches,"Atta packet is inverted",cv::Point(limg.cols,900),cv::FONT_HERSHEY_DUPLEX, 2.0, CV_RGB(255,255,255),2);
            else if((angle>270-angle_range_threshold && angle<270+angle_range_threshold))
            putText(img_Matches,"Left edge of atta packet facing downwards",cv::Point(limg.cols,900),cv::FONT_HERSHEY_DUPLEX, 2.0, CV_RGB(255,255,255),2);


              cout<<"Determinant = "<<determinant<<endl;




                std::cout <<"\n\n Matching and Homography "<< float( clock () - begin_time ) /  CLOCKS_PER_SEC;

                obj_corners[0] = cv::Point2f(0, 0);
                obj_corners[1] = cv::Point2f( (float)limg.cols, 0 );
                obj_corners[2] = cv::Point2f( (float)limg.cols, (float)limg.rows );
                obj_corners[3] = cv::Point2f( 0, (float)limg.rows );

                perspectiveTransform( obj_corners, scene_corners, H);

                //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                line( img_Matches, scene_corners[0] + cv::Point2f((float)limg.cols, 0),
                      scene_corners[1] + cv::Point2f((float)limg.cols, 0), cv::Scalar(0, 255, 0), 4 );
                line( img_Matches, scene_corners[1] + cv::Point2f((float)limg.cols, 0),
                      scene_corners[2] + cv::Point2f((float)limg.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                line( img_Matches, scene_corners[2] + cv::Point2f((float)limg.cols, 0),
                      scene_corners[3] + cv::Point2f((float)limg.cols, 0), cv::Scalar( 0, 255, 0), 4 );
                line( img_Matches, scene_corners[3] + cv::Point2f((float)limg.cols, 0),
                      scene_corners[0] + cv::Point2f((float)limg.cols, 0), cv::Scalar( 0, 255, 0), 4 );

                line( img2, scene_corners[0] ,
                      scene_corners[1] , cv::Scalar(0, 255, 0), 4 );
                line( img2, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
                line( img2, scene_corners[2],scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
                line( img2, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );

            }




    }

    else {
          img_Matches=cv::Mat::ones(std::max(limg.rows,colorFrame.rows),limg.cols+colorFrame.cols,CV_8UC3);
          limg.copyTo(img_Matches(cv::Rect(0,0,limg.cols,limg.rows)));
           colorFrame.copyTo(img_Matches(cv::Rect(limg.cols,0,colorFrame.cols,colorFrame.rows)));

    }

    outputWriter<<img_Matches;

    if(img_Matches.rows>0)
    cv::imshow("Image matches",img_Matches);

    delete rJob;

}


    return EXIT_SUCCESS;
}
