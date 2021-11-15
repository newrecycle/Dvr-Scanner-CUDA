//sys libraries
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
//defines to make sure opencv knows we have cuda
#define HAVE_NVCUVENC
#define HAVE_NVCUVID_HEADER
#define HAVE_OPENCV_NVCUVENC
#define HAVE_OPENCV_CUDA_CODEC
//OpenCV core imports
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
//Nvidia imports
#include <nvcuvid.h>
#include <cuviddec.h>
#include <cuda.h>
#include <nvEncodeAPI.h>
#include <nppcore.h>
#include <npp.h>
#include <cuda_runtime.h>

//Unused imports, might still need later? not sure yet
//#include <opencv2/cudalegacy/NPP_staging.hpp>
//#include <opencv2/cudalegacy.hpp>
//#include <opencv2/cudev/functional/functional.hpp>
//#include <opencv2/cudev/functional/detail/color_cvt.hpp>
//#include <opencv2/cudev/functional/color_cvt.hpp>
//#include <opencv2/cudev/common.hpp>

int main(int argc, const char* argv[]) {
    //INIT cuda driver!
    void* hHandleDriver = nullptr;
    CUresult cuda_res = cuInit(0);
    if (cuda_res != CUDA_SUCCESS) {
        return false;
    }
    else {
        std::cout << "CUDA init: SUCCESS";
    };

    //clock for measuring fps
    cv::TickMeter clock;
    std::vector<double> gpu_times;
    int frames = 0, time = 0;

    //Input and Output file names
    const std::string fname("B:\\Apex Legends\\Apex Legends 2021.07.29 - 03.52.17.03.DVR.mp4");
    const std::string oname("B:\\Apex Legends\\Apex Legends 2021.07.29 - 03.52.17.03.DVR_OUT.mp4");

    //Kernal (5x5 of ones to compare against video)
    cv::Mat kernal = cv::Mat::ones(5, 5, CV_8U);

    //create video reader, get format, create video writer
    cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
    cv::cudacodec::FormatInfo format = d_reader->format();
    int h = format.height;
    int w = format.width;
    cv::VideoWriter d_writer;
    d_writer.fourcc('X', 'V', 'I', 'D');
    //cv::Ptr<cv::cudacodec::VideoWriter> d_writer = cv::cudacodec::createVideoWriter(oname, d_reader->format().displayArea.size(), 60.0);

    //Create both cuda filters for image processing
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> bgsub = cv::cuda::createBackgroundSubtractorMOG2(500, 16.0, false);
    cv::Ptr<cv::cuda::Filter> morph = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernal);
    //bgsub->setShadowThreshold(double(0));
    
    //create GpuMats. not sure if I should initialize? but I will find out.
    cv::cuda::GpuMat c_frame_filt, c_frame_score, c_frame_mask, c_frame_grey, c_frame_rgb;

    //create window for seeing what frames look like
    //cv::namedWindow("GPU", cv::WINDOW_NORMAL);

    //placeholder mat because I cant render GpuMats on a normal window. This bottleneck will be soon to go.
    cv::Mat okay;

    for (;;) {
        //init clock
        clock.reset(); clock.start();
        //get next frame, if no more frames. exit.
        if (!d_reader->nextFrame(c_frame_rgb)) { break; }
        //std::cout << c_frame_rgb.type();
        //std::cout << c_frame_rgb.depth();
        //convert color to grey, apply background subtractorMOG2, apply Morphology ex we defined before
        cv::cuda::cvtColor(c_frame_rgb, c_frame_grey, cv::COLOR_BGRA2GRAY);
        bgsub->apply(c_frame_grey, c_frame_mask);
        morph->apply(c_frame_mask, c_frame_filt);
        //little math to tell us what frames have motion
        //c_frame_filt.download(okay);
        //double score = cv::sum(okay)(0) / (w*h);
        double score = cv::cuda::sum(c_frame_filt)(0) / (w*h);
        //std::cout << score - 0.15;
        //c_frame_score.upload(c_frame_mask);
        //cv::cuda::calcAbsSum(c_frame_filt, c_frame_score, c_frame_filt);
 
        //download frame to cpu
       

        //display frame
        //cv::imshow("GPU", okay);
        //clock and frame shenanigans
        clock.stop();
        gpu_times.push_back(clock.getTimeMilli());
        frames++;

        //if key pressed. exit.
        if (cv::waitKey(3) > 0) {break;}

    }
    std::sort(gpu_times.begin(), gpu_times.end());
    double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
    std::cout << "GPU : Avg : " << gpu_avg << " ms FPS : " << 1000.0 / gpu_avg << " Frames " << frames << std::endl;
    return 0;
}




