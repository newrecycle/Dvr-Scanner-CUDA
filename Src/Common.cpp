#include "Common.h"

gpuThreadManager::gpuThreadManager(std::string fin, std::string fnameout, cv::Mat kern, int threads)
{
    fname = fin; fout = fnameout; kernal = kern; thNum = threads; fNum = 0; ident = 0; 
    d_reader = cv::cudacodec::createVideoReader(fname);
    cv::cudacodec::FormatInfo format = d_reader->format();
    h = format.height; format.width;
    bgsub = cv::cuda::createBackgroundSubtractorMOG2(500, 16.0, false);
    morph = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernal);
    d_writer.fourcc('X', 'V', 'I', 'D');
    //vidBufIn.create(std::vector<int>(5000, 1), CV_8UC4);
}

// I AM CURRENTLY USING VERY UGLY LOCKS TO KEEP THREADS IN FRAME ORDER
// IF YOU KNOW A BETTER WAY TO DO THIS PLEASE CREATE A GITHUB ISSUE!!!

void gpuThreadManager::start()
{   
    cv::cuda::setBufferPoolUsage(true);                           // Tell OpenCV that we are going to utilize BufferPool
    cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), 1024 * 1024 * 256, 10);  // Allocate 64 MB, 2 stacks (default is 10 MB, 5 stacks)
    isDone = false;
    threads[0] = std::thread(&gpuThreadManager::startDecode, this);
    threads[1] = std::thread(&gpuThreadManager::startColorCVT, this);
    threads[2] = std::thread(&gpuThreadManager::startBGsub, this);
    threads[3] = std::thread(&gpuThreadManager::startMorph, this);
    threads[4] = std::thread(&gpuThreadManager::startWriter, this);
    while (isDone == false && vidBufBGsub.size() != 0) {
        Sleep(1);
    }
}
void gpuThreadManager::threadLoop() 
{

}

void gpuThreadManager::startColorCVT()
{
    cv::cuda::GpuMat c_frame_grey;
    while (true)
    {   
        cv::cuda::Stream streamCVT = cv::cuda::Stream();
        cv::cuda::BufferPool poolDec(streamCVT);
        if (vidBufCVT.size() == 0 && vidBufIn.size() == 0 && isDone == true) { break; }
        if (vidBufCVT.size() <= 200 && vidBufIn.size() != 0)
        {
            cv::cuda::cvtColor(vidBufIn[0], c_frame_grey, cv::COLOR_BGRA2GRAY, 0, streamCVT);
            streamCVT.waitForCompletion();
            vidBufInMux.lock();
            vidBufIn.erase(vidBufIn.begin());
            vidBufInMux.unlock();
            vidBufCVTMux.lock();
            vidBufCVT.push_back(c_frame_grey);
            vidBufCVTMux.unlock();
        }
        else {
            Sleep(0);
        }
    }
}

void gpuThreadManager::startBGsub()
{
    cv::cuda::GpuMat c_frame_mask;
    while (true)
    {
        cv::cuda::Stream streamBGsub = cv::cuda::Stream();
        cv::cuda::BufferPool poolDec(streamBGsub);
        if (vidBufBGsub.size() == 0 && vidBufCVT.size() == 0 && isDone == true) { break; }
        if (vidBufBGsub.size() <= 200 && vidBufCVT.size() != 0)
        {
            bgsub->apply(vidBufCVT[0], c_frame_mask, -1.0, streamBGsub);
            
            streamBGsub.waitForCompletion();
            vidBufCVTMux.lock();
            vidBufCVT.erase(vidBufCVT.begin());
            vidBufCVTMux.unlock();
            vidBufBGsubMux.lock();
            vidBufBGsub.push_back(c_frame_mask);
            vidBufBGsubMux.unlock();
        }
        else {
            Sleep(0);
        }
    }
}

void gpuThreadManager::startMorph()
{
    cv::cuda::GpuMat c_frame_filter;
    while (true)
    {
        cv::cuda::Stream streamMorph = cv::cuda::Stream();
        cv::cuda::BufferPool poolDec(streamMorph);
        if (vidBufMorph.size() == 0 && vidBufBGsub.size() == 0 && isDone == true) { break; }
        if (vidBufMorph.size() <= 200 && vidBufBGsub.size() != 0)
        {
            morph->apply(vidBufBGsub[0], c_frame_filter, streamMorph);

            streamMorph.waitForCompletion();
            vidBufBGsubMux.lock();
            vidBufBGsub.erase(vidBufBGsub.begin());
            vidBufBGsubMux.unlock();
            vidBufMorphMux.lock();
            vidBufMorph.push_back(c_frame_filter);
            vidBufMorphMux.unlock();
        }
        else {
            Sleep(0);
        }
    }
}



void gpuThreadManager::startWriter()
{   
    int frames = 0, time = 0;
    clock.start();
    while (true)
    {
        if (vidBufMorph.size() == 0 && vidBufBGsub.size() == 0 && isDone == true) { clock.stop(); 
        std::cout << frames << " Frames in " << clock.getTimeSec() << " Seconds" << std::endl;
        break; }
        if (vidBufMorph.size() != 0) {
            vidBufMorphMux.lock();
            vidBufMorph.pop_back(); 
            vidBufMorphMux.unlock();
            frames++; }
        
        //if (vidBufMorph.size() <= 5000)
        //{
        //    morph->apply(vidBufBGsub[0], c_frame_filter);
        //    vidBufMorph.push_back(c_frame_filter.clone());
        //}
        //else {
        //    Sleep(0);
       // }
    }
}

void gpuThreadManager::startThread()
{
    for (int i = 0; i < 5; i++)
    {
    }
}
void gpuThreadManager::startDecode()
{
    cv::cuda::Stream streamDec = cv::cuda::Stream(CUstream_flags_enum::CU_STREAM_NON_BLOCKING);
    cv::cuda::BufferPool poolDec(streamDec);
    cv::cuda::GpuMat c_frame_rgb = cv::cuda::GpuMat();
    //poolDec.getBuffer(d_reader->format().displayArea.size(), CV_8UC4);
    //poolDec2.getBuffer(d_reader->format().displayArea.size(), CV_8UC4);
    while (true)
    {   
        if (vidBufIn.size() <= 600)
        {
            if (d_reader->nextFrame(c_frame_rgb, streamDec)) {
                streamDec.waitForCompletion();
                vidBufInMux.lock();
                vidBufIn.push_back(c_frame_rgb);
                vidBufInMux.unlock();
            }
            else { isDone = true; break; }
        } else {
            Sleep(0);
        }
    }
}
