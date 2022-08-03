//
// Created by 86176 on 2022/8/1.
//
#include "figure_detection.h"

using namespace cv;
using namespace std;

int main(){
    Mat frame=imread("../SVM_img/7/800.png");

    DWORD Start,end;

    figure_detection detection;
    Start=GetTickCount();
    int out=detection.detect(frame);
    cout<<"out:"<<out<<",confidence"<<detection.confidence<<endl;
    end=GetTickCount();
    cout<<end-Start<<endl;
}


