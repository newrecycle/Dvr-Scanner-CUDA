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



class gpuThreadManager {
private:
    std::string fname = "";
    std::string fout = "";
    cv::Mat kernal;

    std::vector<std::shared_ptr<cv::cuda::GpuMat>>
        vidBufIn,
        vidBufBRG,
        vidBufCVT,
        vidBufBGsub,
        vidBufMorph;
    int
        fullIn,
        fullBRG,
        fullCVT,
        fullBGsub,
        fullMorph;
    
    std::vector<cv::Mat> 
        //vidBufCpu, 
        vidBufOut;
    
    std::array<cv::cuda::GpuMat, 400> allocGpuMat;
    std::array<cv::cuda::GpuMat, 400> allocGpuBRGMat;


    std::mutex 
        vidBufInMux,
        vidBufBRGMux,
        vidBufCVTMux,
        vidBufBGsubMux,
        vidBufMorphMux,
        //vidBufCpuMux,
        vidBufOutMux;
    
    bool inEvent, isDone, isDoneDecode;

    int thNum, h, w, tframes, ident = 0;

    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> bgsub;
    cv::Ptr<cv::cuda::Filter> morph;

    cv::Ptr<cv::cudacodec::VideoReader> d_reader;
    cv::VideoWriter d_writer;
    static cv::cudacodec::FormatInfo format;

    std::thread mainThreads[7];
    std::mutex vidInMtx;
    cv::TickMeter clock;

    void startDecode();
    void startColorCVT();
    void startBGsub();
    void startMorph();
    void startCopyToCpu();
    void startCalculateScore();
    void startWriter();

    bool calculateScore(cv::cuda::GpuMat);
    void moveFrame(std::shared_ptr<cv::cuda::GpuMat> in, std::vector<std::shared_ptr<cv::cuda::GpuMat>> *arr, std::mutex *lock);
    void moveFrame(cv::Mat in, std::vector<cv::Mat> *arr, std::mutex* lock);
    void deleteFirstFrame(std::vector<std::shared_ptr<cv::cuda::GpuMat>> *arr, std::mutex *lock);
public:
    void start();
    gpuThreadManager(std::string, std::string, cv::Mat kern, int threads);
};
#endif