#pragma once

#include <math.h>
#include <time.h>

#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include <net.h>
#include "benchmark.h"

using namespace std;

class DiffusionSlover
{
public:
    DiffusionSlover();

    ncnn::Mat sampler(int seed, int step, ncnn::Mat& c, ncnn::Mat& uc);

private:
    ncnn::Mat randn_4_64_64(int seed);
    ncnn::Mat CFGDenoiser_CompVisDenoiser(ncnn::Mat& input, float sigma, ncnn::Mat cond, ncnn::Mat uncond);

private:
    float log_sigmas[1000] = { 0 };

    ncnn::Net net;
};