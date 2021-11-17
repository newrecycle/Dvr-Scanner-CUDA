#include "Common.h"
#include <Windows.h>
#include <processthreadsapi.h>
#include <thread>
gpuThreadManager::gpuThreadManager(std::string fin, std::string fnameout, cv::Mat kern, int threads)
{
    fname = fin; fout = fnameout; kernal = kern; thNum = threads; fNum = 0; ident = 0; 
    d_reader = cv::cudacodec::createVideoReader(fname);
    
    cv::cudacodec::FormatInfo format = d_reader->format();
    d_writer = cv::VideoWriter(fnameout, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 60, format.displayArea.size());
    h = format.height; format.width;
    bgsub = cv::cuda::createBackgroundSubtractorMOG2(500, 16.0, false);
    morph = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernal);

}

// I AM CURRENTLY USING VERY UGLY LOCKS TO KEEP THREADS IN FRAME ORDER
// IF YOU KNOW A BETTER WAY TO DO THIS PLEASE CREATE A GITHUB ISSUE!!!
void gpuThreadManager::moveFrame(cv::cuda::GpuMat in, std::vector<cv::cuda::GpuMat> *arr, std::mutex *lock) {
    lock->lock();
    arr->push_back(in);
    lock->unlock();
}
void gpuThreadManager::moveFrame(cv::Mat in, std::vector<cv::Mat> *arr, std::mutex *lock) {
    lock->lock();
    arr->push_back(in);
    lock->unlock();
}
void gpuThreadManager::deleteFirstFrame(std::vector<cv::cuda::GpuMat> *arr, std::mutex *lock) {
    lock->lock();
    arr->erase(arr->begin());
    lock->unlock();
}
// I AM CURRENTLY USING VERY UGLY LOCKS TO KEEP THREADS IN FRAME ORDER
// IF YOU KNOW A BETTER WAY TO DO THIS PLEASE CREATE A GITHUB ISSUE!!!

bool gpuThreadManager::calculateScore(cv::Mat temp) {
    double score = cv::sum(temp)(0) / (w*h);
    if (score >= 0.15) {
        return true;
    }
    return false;
}

void gpuThreadManager::start()
{   
    //cv::cuda::setBufferPoolUsage(true);                           // Tell OpenCV that we are going to utilize BufferPool
    //cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), 1024 * 1024 * 128, 5);  // Allocate 256 MB, 10 stacks (default is 10 MB, 5 stacks) because why not lmao
    isDone = false; //we're not done yet!
    isDoneDecode = false; //we're not done yet!
    tframes = 0;
    mainThreads[0] = std::thread(&gpuThreadManager::startDecode, this); //start decode thread
    mainThreads[1] = std::thread(&gpuThreadManager::startColorCVT, this); //start colorCVT thread
    mainThreads[2] = std::thread(&gpuThreadManager::startBGsub, this); //start background subtract thread
    mainThreads[3] = std::thread(&gpuThreadManager::startMorph, this); //start morph thread
    mainThreads[4] = std::thread(&gpuThreadManager::startCopyToCpu, this); //start copying frames that have motion into cpu thread
    mainThreads[5] = std::thread(&gpuThreadManager::startWriter, this); //start cpu encode thread
    clock.start();
    while (isDoneDecode == false || vidBufIn.size() != 0 || vidBufOut.size() != 0 || vidBufBRG.size() != 0) {
        Sleep(1);
    }
    clock.stop();
    
    Sleep(5000);
    isDone = true;
    Sleep(5000);
    for (int i = 0; i <= 5; i++) {
        mainThreads[i].join();
    }
    std::cout << clock.getTimeSec() << " Seconds" << std::endl;
    
    
}

void gpuThreadManager::startDecode() {
    cv::cuda::Stream streamDec = cv::cuda::Stream(CUstream_flags_enum::CU_STREAM_NON_BLOCKING);
    cv::cuda::BufferPool poolDec(streamDec);
    cv::cuda::GpuMat c_frame_rgb = cv::cuda::GpuMat();
    //poolDec.getBuffer(d_reader->format().displayArea.size(), CV_8UC4);
    //poolDec2.getBuffer(d_reader->format().displayArea.size(), CV_8UC4);
    while (true) {
        if (vidBufIn.size() <= 150 && vidBufBRG.size() <= 400) {
            if (d_reader->nextFrame(c_frame_rgb, streamDec)) {
                streamDec.waitForCompletion();
                moveFrame(c_frame_rgb, &vidBufIn, &vidBufInMux);
                moveFrame(c_frame_rgb.clone(), &vidBufBRG, &vidBufBRGMux);
                tframes++;
            } else { 
                isDoneDecode = true; 
                std::cout << std::endl << tframes << " Frames in ";
                break; }
        } else {
            Sleep(0);
        }
    }
}

void gpuThreadManager::startColorCVT() {
    cv::cuda::GpuMat c_frame_grey;
    cv::cuda::Stream streamCVT = cv::cuda::Stream(CUstream_flags_enum::CU_STREAM_NON_BLOCKING);
    cv::cuda::BufferPool poolDec(streamCVT);
    int colF = 0;
    while (true) {   
        if (vidBufCVT.size() == 0 && vidBufIn.size() == 0 && isDone == true) { 
            std::cout << colF << "COLORCVT Frames";
            break; }
        if (vidBufCVT.size() <= 200 && vidBufIn.size() != 0) {
            cv::cuda::cvtColor(vidBufIn[0], c_frame_grey, cv::COLOR_BGRA2GRAY, 0, streamCVT);
            streamCVT.waitForCompletion();
            moveFrame(c_frame_grey, &vidBufCVT, &vidBufCVTMux);
            deleteFirstFrame(&vidBufIn, &vidBufInMux);
            colF++;
        }
        else {
            Sleep(0);
        }
    }
}

void gpuThreadManager::startBGsub() {
    cv::cuda::GpuMat c_frame_mask;
    cv::cuda::Stream streamBGsub = cv::cuda::Stream(CUstream_flags_enum::CU_STREAM_NON_BLOCKING);
    cv::cuda::BufferPool poolDec(streamBGsub);
    int bgsF = 0;
    while (true) {
        if (vidBufBGsub.size() == 0 && vidBufCVT.size() == 0 && isDone == true) { 
            std::cout << bgsF << "BGSUB Frames";
            break;}
        if (vidBufBGsub.size() <= 200 && vidBufCVT.size() != 0) {
            bgsub->apply(vidBufCVT[0], c_frame_mask, -1.0, streamBGsub);
            streamBGsub.waitForCompletion();
            deleteFirstFrame(&vidBufCVT, &vidBufCVTMux);
            moveFrame(c_frame_mask, &vidBufBGsub, &vidBufBGsubMux);
            bgsF++;
        } else {
            Sleep(0);
        }
    }
}
void gpuThreadManager::startMorph() {
    cv::cuda::GpuMat c_frame_filter;
    cv::cuda::Stream streamMorph = cv::cuda::Stream(CUstream_flags_enum::CU_STREAM_NON_BLOCKING);
    cv::cuda::BufferPool poolDec(streamMorph);
    int morF = 0;
    while (true) {
        if (vidBufMorph.size() == 0 && vidBufBGsub.size() == 0 && isDone == true) { 
            std::cout << morF << "BGSUB Frames";
            break;}
        if (vidBufMorph.size() <= 200 && vidBufBGsub.size() != 0) {
            morph->apply(vidBufBGsub[0], c_frame_filter, streamMorph);
            streamMorph.waitForCompletion();
            deleteFirstFrame(&vidBufBGsub, &vidBufBGsubMux);
            moveFrame(c_frame_filter, &vidBufMorph, &vidBufMorphMux);
            morF++;
        } else {
            Sleep(0);
        }
    }
}
void gpuThreadManager::startCopyToCpu() {
    cv::Mat temp, temp2;
    inEvent = false;
    int inEventCount = 0;
    while (true) {
        if (vidBufBGsub.size() == 0 && vidBufMorph.size() == 0 && isDone == true) {
            break;
        }
        if (vidBufMorph.size() != 0) {
            vidBufMorph[0].download(temp);
            if (calculateScore(temp)) {
                inEvent = true;
                inEventCount = 120;
            }
            if (inEvent) {
                if (inEventCount == 0) {
                    inEvent = false;
                }
                else {
                    vidBufBRGMux.lock();
                    vidBufBRG[0].download(temp2);
                    vidBufBRGMux.unlock();
                    moveFrame(temp2, &vidBufOut, &vidBufOutMux);
                    inEventCount--;
                }
            }
            deleteFirstFrame(&vidBufMorph, &vidBufMorphMux);
            deleteFirstFrame(&vidBufBRG, &vidBufBRGMux);
        }
        else {
            Sleep(0);
        }
    }
}

void gpuThreadManager::startWriter() {   
    int frames = 0, time = 0;
    
    cv::Mat temp;
    while (true) {
        if (vidBufMorph.size() == 0 && vidBufBGsub.size() == 0 && vidBufOut.size() == 0 && isDone == true) { 
            break; 
        }
        if (vidBufOut.size() != 0) {
            vidBufOutMux.lock();
            cv::cvtColor(vidBufOut[0], temp, cv::COLOR_BGRA2BGR);
            d_writer.write(temp);
            vidBufOut.erase(vidBufOut.begin());
            vidBufOutMux.unlock();
            frames++;
        }
    }
}
