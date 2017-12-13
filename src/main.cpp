
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp> 
#include <sstream>
#include <fstream>
using namespace cv;
using namespace std;

char * imgname = "../img1.jpg";

void bgr2BW(Mat bgrimg,Mat * BW_res);
void img_preview(Mat img);

int main(int argc, char** argv)
{
	Mat raw_img = imread(imgname);
    Mat bw_img;
    namedWindow("raw",CV_WINDOW_NORMAL);
    imshow("raw",raw_img);
    bgr2BW(raw_img,&bw_img);

    //cout<<raw_img<<endl;
    waitKey(0);
 	return 0; 
}

void bgr2BW(Mat bgrimg,Mat * BW_res)
{
    Mat channel[3];  
    Mat E(bgrimg.size(),CV_8UC1,Scalar(255));
    Mat BW(bgrimg.size(),CV_8UC1,Scalar(0));
    Mat tmp;
    img_preview(BW);
    img_preview(E);

    split(bgrimg,channel); 
    for(int i=2;i<3;i++)
    {
    tmp=channel[i].clone();

    img_preview(tmp);

    threshold(tmp, tmp, 25, 255.0, CV_THRESH_BINARY);
    
    img_preview(tmp);
    Mat ele = getStructuringElement(MORPH_RECT,Size(3,3));//get kernel
    erode(tmp,tmp,ele);

    bitwise_or(tmp,BW,tmp);

    img_preview(tmp);




    }
}


void img_preview(Mat img)
{
    imshow("img_preview",img);
    waitKey(0);
}

