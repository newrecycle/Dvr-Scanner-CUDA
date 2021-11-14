#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#define HAVE_NVCUVENC
//#undef _WIN32
#define HAVE_NVCUVID_HEADER
#define HAVE_OPENCV_NVCUVENC
#define HAVE_OPENCV_CUDA_CODEC
#define const static enum HAVE_NVCUVENC;
#define const static enum HAVE_NVCUVID_HEADER;
#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#include <nvcuvid.h>
#include <cuviddec.h>
#include <cuda.h>
#include <nvEncodeAPI.h>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/bgsegm.hpp>
//#include <opencv2/cudalegacy/NPP_staging.hpp>
//#include <opencv2/cudalegacy.hpp>
#include <nppcore.h>
#include <npp.h>
#include <cuda_runtime.h>
//#include <opencv2/cudev/functional/functional.hpp>
//#include <opencv2/cudev/functional/detail/color_cvt.hpp>
//#include <opencv2/cudev/functional/color_cvt.hpp>
//#include <opencv2/cudev/common.hpp>
#undef _WIN32
#define HAVE_OPENCV_NVCUVENC
#define HAVE_OPENCV_CUDA_CODEC
int main(int argc, const char* argv[]) {
    const static enum HAVE_OPENCV_NVUCENV;

    const static enum HAVE_NVCUVENC;
    const static enum HAVE_NVCUVID_HEADER;
    const static enum HAVE_OPENCV_NVUCENV;
    void* hHandleDriver = nullptr;
    CUresult cuda_res = cuInit(0);
    if (cuda_res != CUDA_SUCCESS) {
        return false;
    }
    else {
        std::cout << "CUDA init: SUCCESS";
    };
    const std::string fname("B:\\Apex Legends\\Apex Legends 2021.07.29 - 03.52.17.03.DVR.mp4");
    const std::string oname("B:\\Apex Legends\\Apex Legends 2021.07.29 - 03.52.17.03.DVR_OUT.mp4");
    cv::cudacodec::ChromaFormat::YUV420;
    cv::Mat kernal = cv::Mat::ones(5, 5, CV_8U);
    cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
    d_reader->format().displayArea.size().height;
    cv::Ptr<cv::cudacodec::VideoWriter> d_writer = cv::cudacodec::createVideoWriter(oname, d_reader->format().displayArea.size(), 60.0);
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> bgsub = cv::cuda::createBackgroundSubtractorMOG2(500, 16.0, false);
    cv::Ptr<cv::cuda::Filter> morph = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernal);
    //morph->;
    //cv::Pts<cv::cuda
    //bgsub->setShadowThreshold(double(0));
    cv::cuda::GpuMat c_frame_filt, c_frame_score, c_frame_mask, c_frame_grey;
    //cv::Mat ;
    int h = d_reader->format().displayArea.height;
    int w = d_reader->format().displayArea.width;
    //cv::cuda::GpuMat c_frame_rgb = cv::cuda::GpuMat(w, h, cv::cudacodec::ChromaFormat::YUV420);
    cv::cuda::GpuMat c_frame_rgb;
    cv::namedWindow("GPU", cv::WINDOW_NORMAL);
    cv::Mat okay;
    for (;;) {
        if (!d_reader->nextFrame(c_frame_rgb)) { break; }
        //std::cout << c_frame_rgb.type();
        //std::cout << c_frame_rgb.depth();
        
        //NppStatus status = nppiNV12ToBGR_709HDTV_8u_P2C3R(cv::, c_frame_rgb.step);
        cv::cuda::cvtColor(c_frame_rgb, c_frame_grey, cv::COLOR_BGRA2GRAY);
        bgsub->apply(c_frame_grey, c_frame_mask);
        morph->apply(c_frame_mask, c_frame_filt);
        double score = cv::cuda::sum(c_frame_filt)(0) / double(w * h);
        std::cout << score - 0.15;
        //c_frame_score.upload(c_frame_mask);
        //cv::cuda::calcAbsSum(c_frame_filt, c_frame_score, c_frame_filt);
        c_frame_filt.download(okay);
        cv::imshow("GPU", okay);
        if (cv::waitKey(3) > 0)
            break;
    }
}




