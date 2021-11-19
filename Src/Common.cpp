#include "Common.h"

gpuThreadManager::gpuThreadManager(std::string fin, std::string fnameout, cv::Mat kern, int threads)
{
    fname = fin; fout = fnameout; kernal = kern; thNum = threads; ident = 0;
    cv::cuda::GpuMat firstFrame;
    d_reader = cv::cudacodec::createVideoReader(fname);
    d_reader->nextFrame(firstFrame);
    cv::cudacodec::FormatInfo format = d_reader->format();
    d_writer = cv::VideoWriter(fnameout, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 60, format.displayArea.size());
    allocGpuMat.fill(cv::cuda::GpuMat(firstFrame));
    allocGpuBRGMat.fill(cv::cuda::GpuMat(firstFrame));
    h = format.height; format.width;
    bgsub = cv::cuda::createBackgroundSubtractorMOG2(500, 16.0, false);
    morph = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernal);

}

// I AM CURRENTLY USING VERY UGLY LOCKS TO KEEP THREADS IN FRAME ORDER
// IF YOU KNOW A BETTER WAY TO DO THIS PLEASE CREATE A GITHUB ISSUE!!!
void gpuThreadManager::moveFrame(std::shared_ptr<cv::cuda::GpuMat> in, std::vector<std::shared_ptr<cv::cuda::GpuMat>> *arr, std::mutex *lock) {
    lock->lock();
    arr->push_back(in);
    lock->unlock();
}
void gpuThreadManager::moveFrame(cv::Mat in, std::vector<cv::Mat> *vec, std::mutex *lock) {
    lock->lock();
    vec->push_back(in);
    lock->unlock();
}
void gpuThreadManager::deleteFirstFrame(std::vector<std::shared_ptr<cv::cuda::GpuMat>> *arr, std::mutex *lock) {
    lock->lock();
    arr->erase(arr->begin());
    lock->unlock();
}
// I AM CURRENTLY USING VERY UGLY LOCKS TO KEEP THREADS IN FRAME ORDER
// IF YOU KNOW A BETTER WAY TO DO THIS PLEASE CREATE A GITHUB ISSUE!!!

bool gpuThreadManager::calculateScore(cv::cuda::GpuMat temp) {
    double score = cv::cuda::sum(temp)(0.0f) / double(w*h);
    if (score >= 0.15) {
        return true;
    }
    return false;
}

void gpuThreadManager::start()
{   
    //cv::cuda::setBufferPoolUsage(true);                           // Tell OpenCV that we are going to utilize BufferPool
    //cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), 1024 * 1024 * 256, 5);  // Allocate 256 MB, 10 stacks (default is 10 MB, 5 stacks) because why not lmao
    isDone = false; //we're not done yet!
    isDoneDecode = false; //we're not done yet!
    tframes = 0;
    mainThreads[0] = std::thread(&gpuThreadManager::startDecode, this); //start decode thread
    mainThreads[1] = std::thread(&gpuThreadManager::startColorCVT, this); //start colorCVT thread
    mainThreads[2] = std::thread(&gpuThreadManager::startBGsub, this); //start background subtract thread
    mainThreads[3] = std::thread(&gpuThreadManager::startMorph, this); //start morph thread
    //mainThreads[4] = std::thread(&gpuThreadManager::startCopyToCpu, this); //start copying frames that have motion into cpu thread
    mainThreads[4] = std::thread(&gpuThreadManager::startCalculateScore, this);
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
    std::cout << std::endl << fullIn << " In";
    std::cout << std::endl << fullBRG << " Brg";
    std::cout << std::endl << fullCVT << " CVT";
    std::cout << std::endl << fullBGsub << " BGsub";
    std::cout << std::endl << fullMorph << " Morph";
    
    
}

void gpuThreadManager::startDecode() {
    cv::cuda::Stream streamDec = cv::cuda::Stream(CUstream_flags_enum::CU_STREAM_NON_BLOCKING);
    cv::cuda::BufferPool poolDec(streamDec);
    cv::cuda::Stream streamCpy = cv::cuda::Stream(CUstream_flags_enum::CU_STREAM_NON_BLOCKING);
    cv::cuda::GpuMat c_frame_rgb = cv::cuda::GpuMat();
    std::thread cp([&] {; }), cp1([&] {; }), cp2([&] {; });
    //poolDec.getBuffer(d_reader->format().displayArea.size(), CV_8UC4);
    //poolDec2.getBuffer(d_reader->format().displayArea.size(), CV_8UC4);
    while (true) {
        int i = 0;
        while (i < 400) {
            if (vidBufIn.size() <= 100) {
                if (vidBufBRG.size() < 399) {
                    if (d_reader->nextFrame(allocGpuMat[i], streamDec)) {
                        streamDec.waitForCompletion();
                        std::thread cp([&, this] {moveFrame(std::make_shared<cv::cuda::GpuMat>(allocGpuMat[i]), &vidBufIn, &vidBufInMux); });
                        std::thread cp1([&, this] {allocGpuMat[i].copyTo(allocGpuBRGMat[i], streamCpy);});
                        //moveFrame(std::make_shared<cv::cuda::GpuMat>(allocGpuMat[i]), &vidBufIn, &vidBufInMux);
                        streamCpy.waitForCompletion();
                        std::thread cp2([&, this] {moveFrame(std::make_shared<cv::cuda::GpuMat>(allocGpuMat[i]), &vidBufBRG, &vidBufBRGMux); });
                        //moveFrame(std::make_shared<cv::cuda::GpuMat>(allocGpuMat[i + 399]), &vidBufBRG, &vidBufBRGMux);
                        cp.join(); cp1.join(); cp2.join();
                        tframes++;
                        i++; 
                    } else {
                        isDoneDecode = true;
                        std::cout << std::endl << tframes << " Frames in ";
                        break;
                    }
                } else {
                    Sleep(0);
                    fullBRG++;
                }
            } else {
                Sleep(0);
                fullIn++;
            }
        }
        if (isDoneDecode) { break; }
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
        if (vidBufCVT.size() <= 100 && vidBufIn.size() != 0) {
            vidBufInMux.lock();
            cv::cuda::cvtColor(*vidBufIn[0], *vidBufIn[0], cv::COLOR_BGRA2GRAY, 0, streamCVT);
            streamCVT.waitForCompletion();
            vidBufInMux.unlock();
            moveFrame(vidBufIn[0], &vidBufCVT, &vidBufCVTMux);
            deleteFirstFrame(&vidBufIn, &vidBufInMux);
            colF++;
        } else {
            Sleep(0);
            fullCVT++;
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
        if (vidBufBGsub.size() <= 100 && vidBufCVT.size() != 0) {
            vidBufCVTMux.lock();
            bgsub->apply(*vidBufCVT[0], *vidBufCVT[0], -1.0, streamBGsub);
            streamBGsub.waitForCompletion();
            vidBufCVTMux.unlock();
            moveFrame(vidBufCVT[0], &vidBufBGsub, &vidBufBGsubMux);
            deleteFirstFrame(&vidBufCVT, &vidBufCVTMux);
            
            bgsF++;
        } else {
            Sleep(0);
            fullBGsub++;
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
            std::cout << morF << "Morph Frames";
            break;}
        if (vidBufMorph.size() <= 100 && vidBufBGsub.size() != 0) {
            vidBufBGsubMux.lock();
            morph->apply(*vidBufBGsub[0], *vidBufBGsub[0], streamMorph);
            streamMorph.waitForCompletion();
            vidBufBGsubMux.unlock();
            moveFrame(vidBufBGsub[0], &vidBufMorph, &vidBufMorphMux);
            deleteFirstFrame(&vidBufBGsub, &vidBufBGsubMux);
            morF++;
        } else {
            Sleep(0);
            fullMorph++;
        }
    }
}
void gpuThreadManager::startCopyToCpu() {
    //cv::Mat temp;
    
    /*while (true) {
        if (vidBufBGsub.size() == 0 && vidBufMorph.size() == 0 && isDone == true) {
            break;
        }
        if (vidBufMorph.size() != 0) {
            vidBufMorphMux.lock();
            vidBufMorph[0]->download(temp);
            vidBufMorphMux.unlock();
            moveFrame(temp, &vidBufCpu, &vidBufCpuMux);
            
            deleteFirstFrame(&vidBufMorph, &vidBufMorphMux);
        } else {
            Sleep(0);
        }
    }*/
}

void gpuThreadManager::startCalculateScore() {
    inEvent = false;
    int inEventCount = 0;
    cv::Mat temp;
    cv::cuda::Stream streamCalc = cv::cuda::Stream(CUstream_flags_enum::CU_STREAM_NON_BLOCKING);
    while (true) {
        if (vidBufMorph.size() == 0 && isDone == true) {
            break;
        }
        if (vidBufMorph.size() != 0) {
            if (calculateScore(*vidBufMorph[0])) {
                inEvent = true;
                inEventCount = 120;
            }
            if (inEvent) {
                if (inEventCount == 0) {
                    inEvent = false;
                } else {
                    //vidBufBRGMux.lock();
                    vidBufBRG[0]->download(temp, streamCalc);
                    deleteFirstFrame(&vidBufMorph, &vidBufMorphMux);
                    //vidBufBRGMux.unlock();
                    streamCalc.waitForCompletion();
                    moveFrame(temp, &vidBufOut, &vidBufOutMux);
                    inEventCount--;
                    deleteFirstFrame(&vidBufBRG, &vidBufBRGMux);
                }
            }
            else {
                deleteFirstFrame(&vidBufMorph, &vidBufMorphMux);
                deleteFirstFrame(&vidBufBRG, &vidBufBRGMux);
            }
            
        } else {
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
