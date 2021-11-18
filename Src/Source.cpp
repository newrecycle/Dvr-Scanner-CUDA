#include "Common.h"
#include "Source.h"

bool initDriver() {
    std::cout << "Initializing Cuda" << std::endl;
    void* hHandleDriver = nullptr;
    CUresult cuda_res = cuInit(0);
    if (cuda_res == CUDA_SUCCESS) {
        std::cout << "CUDA init: PASS" << std::endl;
        return true;
    }
    else {
        std::cout << "CUDA init: FAIL" << std::endl;
        return false;
    };
}
int main() {

    initDriver();

    //clock for measuring fps
    cv::TickMeter clock;
    std::vector<double> gpu_times;
    int frames = 0, time = 0;

    //Input and Output file names
    const std::string fName("B:\\2021-11-04 17-36-48.mp4");
    const std::string oName("B:\\Yo_ForReal.avi");

    //Kernal (5x5 of ones to compare against video)
    cv::Mat kernal = cv::Mat::ones(5, 5, CV_8U);

    //create video reader, get format, create video writer
    
    
    //cv::Ptr<cv::cudacodec::VideoWriter> d_writer = cv::cudacodec::createVideoWriter(oname, d_reader->format().displayArea.size(), 60.0);

    //Create both cuda filters for image processing
    //bgsub->setShadowThreshold(double(0));

    //create window for seeing what frames look like
    //cv::namedWindow("GPU", cv::WINDOW_NORMAL);

    //placeholder mat because I cant render GpuMats on a normal window. This bottleneck will be soon to go.

    gpuThreadManager gpuT(fName, oName, kernal, 2);
    gpuT.start();
    std::cout << std::endl << "Exited threadManagerLoop" << std::endl;
    Sleep(5000);
    return 0;
}
    
    //for (;;) {
        //init clock
        //clock.reset(); clock.start();
        //get next frame, if no more frames. exit.
        //gpuT.startDecode();
        //std::cout << c_frame_rgb.type();
        //std::cout << c_frame_rgb.depth();
        //convert color to grey, apply background subtractorMOG2, apply Morphology ex we defined before
        //cv::cuda::cvtColor(c_frame_rgb, c_frame_grey, cv::COLOR_BGRA2GRAY);
        //bgsub->apply(c_frame_grey, c_frame_mask);
        //NewFunction(morph, c_frame_mask, c_frame_filt);
        //little math to tell us what frames have motion
        //c_frame_filt.download(okay);
        //double score = cv::sum(okay)(0) / (w*h);
        //double score = cv::cuda::sum(c_frame_filt)(0) / (w*h);
        //std::cout << score - 0.15;
        //c_frame_score.upload(c_frame_mask);
        //cv::cuda::calcAbsSum(c_frame_filt, c_frame_score, c_frame_filt);
 
        //download frame to cpu
       

        //display frame
        //cv::imshow("GPU", okay);
        //clock and frame shenanigans
        //clock.stop();
        //gpu_times.push_back(clock.getTimeMilli());
        //frames++;

        //if key pressed. exit.
        //if (cv::waitKey(3) > 0) {break;}

    //}
    //std::sort(gpu_times.begin(), gpu_times.end());
    //double gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / gpu_times.size();
    //std::cout << "GPU : Avg : " << gpu_avg << " ms FPS : " << 1000.0 / gpu_avg << " Frames " << frames << std::endl;
    




