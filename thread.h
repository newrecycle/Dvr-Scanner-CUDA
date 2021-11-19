
#ifndef THREAD_H
#define THREAD_H
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
class thread
{
	int thredNum;
	std::thread me;
	std::function<void> func;
	std::arg args[];
};

class gpuThread : public thread
{

};

class cpuThread : public thread
{

};

#endif