//
// Created by 86176 on 2022/8/3.
//

#include "figure_detection.h"

using namespace dnn;
using namespace std;
figure_detection::figure_detection() {
    net= readNetFromONNX("../model_nin.onnx");
    cout<<"read successful"<<endl;
    if(net.empty()){
        printf("Could not load net..\n");
    }
}
Mat figure_detection::pretreat(Mat &Image) {
    Mat gray;
    cvtColor(Image,gray,COLOR_BGR2GRAY);
    //转换为模型需要的张量格式
    Mat inputBlob= blobFromImage(gray,1);
    return inputBlob;
}
int figure_detection::detect(Mat& Image){
    Mat input= pretreat(Image);

    //设置在opencv上推理
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(DNN_TARGET_CUDA);//cuda推理
    //模型输入
    net.setInput(input);
    //模型推理
    vector<float> detectionMat = net.forward();
    //输出最大值对应序号
    int num=0;
    for(int i=0;i<detectionMat.size();i++) {
        //cout<<detectionMat[i]<<endl;
        if (detectionMat[i] > detectionMat[num])
            num = i;
    }
    //置信度
    vector<float> softmax_detect;
    float softmax_sum=0;
    for(int i=0;i<detectionMat.size();i++){
        softmax_detect.push_back(detectionMat[i]/255.F);
        softmax_sum+=exp(softmax_detect[i]);
    }
    confidence=exp(softmax_detect[num])/softmax_sum;
    return num;
}
