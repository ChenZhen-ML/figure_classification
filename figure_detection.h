//
// Created by 86176 on 2022/8/3.
//

#ifndef ML_DETECTION_FIGURE_DETECTION_H
#define ML_DETECTION_FIGURE_DETECTION_H


#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <windows.h>

using namespace cv;
class figure_detection {
private:
    Mat pretreat(Mat& Image);//图片预处理
    dnn::Net net;
public:
    figure_detection();//初始化
    int detect(Mat& Image);//返回图片类别序号
    float confidence;
};


#endif //ML_DETECTION_DIGITAL_DETECTION_H
