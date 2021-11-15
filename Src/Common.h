#ifndef COMMON_H
#define COMMON_H
#include <nvcuvid.h>
#include <cuviddec.h>
//std
#include <iostream>
#include <thread>
#include <mutex>
//opencv
#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/bgsegm.hpp>
//OpenCV cuda imports
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif


//INIT cuda driver!



class gpuThreadManager
{
public:
    std::string fname = "";
    std::string fout = "";
    cv::Mat kernal;
    std::vector<cv::cuda::GpuMat> vidBufIn;
    std::vector<cv::cuda::GpuMat> vidBufCVT;
    std::vector<cv::cuda::GpuMat> vidBufBGsub;
    std::vector<cv::cuda::GpuMat> vidBufMorph;
    std::vector<cv::cuda::GpuMat> vidBufOut;

    std::mutex vidBufInMux;
    std::mutex vidBufCVTMux;
    std::mutex vidBufBGsubMux;
    std::mutex vidBufMorphMux;
    std::mutex vidBufOutMux;
    
    

    int thNum;
    int ident = 0;
    long long fNum = 0;

    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> bgsub;
    cv::Ptr<cv::cuda::Filter> morph;

    cv::Ptr<cv::cudacodec::VideoReader> d_reader;
    cv::VideoWriter d_writer;
    static cv::cudacodec::FormatInfo format;
     bool isDone;
    int h;
    int w;
    std::thread threads[5];
    std::mutex vidInMtx;
    cv::TickMeter clock;
    gpuThreadManager(std::string, std::string, cv::Mat kern, int threads);
    //~gpuThreadManager(void);
    void start();
    void startThread();
    void threadLoop();

    void startDecode();
    void startColorCVT();
    void startBGsub();
    void startMorph();
    void startWriter();
    
};
#endif