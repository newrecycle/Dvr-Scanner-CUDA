
#ifndef THREAD_H
#define THREAD_H
#include <nvcuvid.h>
#include <cuviddec.h>
#include "Src/Common.h"
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


class gpuThread {
	//private:
	int frameNum; //frame number
	int threadNum; //thread number
	int w, h;
	cv::Ptr<cv::cuda::Stream> stream; //stream pointer
	std::array<cv::Ptr<cv::cuda::GpuMat>, 5> *in; // input & output GpuMat pointer
	cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> *bgsub; //BGSub filter pointer
	cv::Ptr<cv::cuda::Filter> *morph; //MorphEx filter pointer
	std::array<cv::Ptr<cv::cuda::GpuMat>, 5> *out;
	
	public:
		void initThread(
			int tNum, //thread number
			int w,
			int h,
			cv::Ptr<cv::cuda::Stream> stm, //stream pointer
			cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> bgs,  //BGSub filter pointer
			cv::Ptr<cv::cuda::Filter> mor //MorphEx filter pointer
		);
		void start(std::array<cv::Ptr<cv::cuda::GpuMat>, 5>& in, std::array<cv::Ptr<cv::cuda::GpuMat>, 5>& out, int fNum);

};

class cpuThread {

};



#endif