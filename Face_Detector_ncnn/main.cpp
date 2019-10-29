#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "FaceDetector.h"

using namespace std;

int main(int argc, char** argv){
    string param = "../model/face.param";
    string bin = "../model/face.bin";
    const int max_side = 320;

    Detector detector(param, bin);
    Timer timer;

	//定义VideoCapture对象选择摄像头
	cv::VideoCapture capture(0);
	//判断是否出错
	if (!capture.isOpened())
	{
		cout << "some thing wrong" << endl;
		system("pause");
		return -1;
	}
	//获取视频相关信息---分辨率（宽、高）
	//int  frameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	//int frameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	//cout << "this video is :" << frameWidth << "*" << frameHeight << endl;
	//定义writer对象

	cv::Mat img;
	while (1) {

		capture >> img;
        // scale
        float long_side = std::max(img.cols, img.rows);
        float scale = max_side/long_side;
        cv::Mat img_scale;
        cv::Size size = cv::Size(img.cols*scale, img.rows*scale);
        cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));

        if (img.empty())
        {
            fprintf(stderr, "cv::imread failed\n");
            return -1;
        }
        std::vector<bbox> boxes;
        timer.tic();
        detector.Detect(img_scale, boxes);
        timer.toc("----total timer:");

        // draw image
        for (int j = 0; j < boxes.size(); ++j) {
            cv::Rect rect(boxes[j].x1/scale, boxes[j].y1/scale, boxes[j].x2/scale - boxes[j].x1/scale, boxes[j].y2/scale - boxes[j].y1/scale);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            char test[80];
            sprintf(test, "%f", boxes[j].s);
            cv::putText(img, test, cv::Size((boxes[j].x1/scale), boxes[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
            cv::circle(img, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(img, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(img, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
        }
        cv::imshow("test.png", img);
		cv::waitKey(1);
    }
    return 0;
}

