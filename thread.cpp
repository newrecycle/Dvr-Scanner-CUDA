#include "thread.h"


void gpuThread::initThread(
	int tNum,
	int width, 
	int height,
	cv::Ptr<cv::cuda::Stream> stm, 
	cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> bgs, 
	cv::Ptr<cv::cuda::Filter> mor
	)

{
	threadNum = tNum;
	w = width;
	h = height;
	stream = &stm;
	bgsub = &bgs;
	morph = &mor;
	
}

void gpuThread::start(std::array<cv::Ptr<cv::cuda::GpuMat>, 5> &in, std::array<cv::Ptr<cv::cuda::GpuMat>, 5> &out, int fNum)
{
	cv::cuda::cvtColor(*in[0], *out[1], cv::COLOR_BGRA2GRAY, 0, *stream);
	//bgsub->apply(*in[1], *out[2], -1.0, streamBGsub);
	(*morph)->apply(*in[2], *out[3], *stream);
	double score = double(cv::cuda::sum(*in[4])(0)) / double(w * h);
	stream->waitForCompletion();
	//return std::array<double, 2>{ score, double(fNum) };

}
