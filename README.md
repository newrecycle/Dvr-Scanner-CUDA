# Dvr-Scanner-CUDA

I love the program Dvr-Scan for going through my cctv footage at a later time. The problem with it is, it only runs at about 230 fps on my 1080p 60fps footage encoded in H264. Keep in mind that is running single threaded on my AMD Ryzen 9 3900XT 24 thread processer. The traditional program and my C++ rewrite both use open cv. The difference with my application is that: OpenCV is built with NVIDIA Cuda support. This allows me to do everything except video encoding with my systems GPU instead of the CPU. Preliminary results show that my RTX 2060 Super can do it around ten time faster (2300fps). This makes going through all of my footage so much faster and easier. The GitHub repo is here: (https://github.com/newrecycle/dvr_scanner_CUDA/tree/main). I am not that far along in my progress but so far I have figured out how to enable CUDA 11.5 (arch 7.5) toolkit support for video reading and image processing. Let me know what you guys think and feel free to go check out my progress & even contribute if you'd like! I am very open to constructive criticism. THIS IS NOT A PROMOTION FOR THIS PROJECT. I simply want to let people know that this program will continue to live and that it will be so much better to use on modern hardware than the old one. r/programming seems to be the best place to do that with likeminded individuals! thanks! 

dvr scanner rewritten in c++.

this program will require opencv built with cuda. 
I have chosen to install all nvidia toolkit drivers, nvcuvid, cudnn, cuviddec and nvEncodeAPI.
nvEncodeAPI will most likely not be used.
