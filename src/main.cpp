
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp> 
#include <sstream>
#include <fstream>
#include <math.h>
using namespace cv;
using namespace std;

char * imgname = "../img2.jpg";

void bgr2BW(Mat bgrimg,Mat * BW_res);
void img_preview(Mat img);
int Median(double a[],int N);

int main(int argc, char** argv)
{
	Mat raw_img = imread(imgname);
    Mat bw_img;
    namedWindow("raw",CV_WINDOW_NORMAL);
    imshow("raw",raw_img);
    bgr2BW(raw_img,&bw_img);
    int i;
    double areaSizeQuene[100];
    int areaSizeCnt=0;
    int delete_flag=0;
    //img_preview(bw_img);
    imshow("bw",bw_img);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Mat blank=Mat::zeros(raw_img.size(),CV_8UC1);
    //bitwise_not(bw_img,bw_img);//inverse
    findContours( bw_img, contours, hierarchy,
        CV_RETR_CCOMP,CV_CHAIN_APPROX_NONE);

    //ite
    

    Moments mu;
    double hu[7];
    double M1,M2,M3,M7;
    vector <int>::iterator Iter;  

    cout<<contours.size()<<endl;
    int ii=0;
    for (vector<vector<Point> >::iterator it = contours.begin(); it != contours.end();) 
    {
        //method of moments
        /*
        mu=moments(*it,true);
        HuMoments( mu,hu);
        double m00=mu.m00;
        M1=(hu[0]);//(pow(m00,2));
        M2=(hu[1]);//(pow(m00,4));
        M3=(hu[2]);//(pow(m00,5));
        M7=(hu[6]);//(pow(m00,4));

        cout<<m00<<endl;
        //cout<<"delete"<<i<<endl;
        */
        delete_flag=0;
        double area=contourArea(*it);
        
        
        double hull_area;
        vector<Point> hull; 
        convexHull(Mat(*it), hull,false,false);
        hull_area=contourArea(hull);
        double Solidity;
        Solidity=area/hull_area;
        //cout<<"Solidity "<<Solidity<<endl;
        if(Solidity<0.85)
        {
            delete_flag=1;
        }

        if(area>5)
        {
            RotatedRect box = fitEllipse(*it);  
            //if()
            if(MIN(box.size.width, box.size.height)/MAX(box.size.width, box.size.height)<0.15)  
            {
                delete_flag=1;
            }
        }

        if(area<200||delete_flag)
        {
            //cout<<"delete "<<area<<endl;
            it=contours.erase(it);
        }
        else
        {
            //cout<<"delete "<<area<<endl;
            areaSizeQuene[areaSizeCnt]=area;
            areaSizeCnt++;
            it++;
        }
        
    }
    
    double medSize;
    medSize=Median(areaSizeQuene,contours.size());
    
    cout<<"medSize :"<<medSize<<endl;

    for (vector<vector<Point> >::iterator it = contours.begin(); it != contours.end();) 
    {
        double area=contourArea(*it);

        if((area<medSize/3)||(area>medSize*3))
        {
            it=contours.erase(it);
        }
        else
        {
            it++;
        }

    }

    cout<<contours.size()<<endl;
    for(i=0; i<contours.size(); i++ )
    {
        Scalar color=255;
        drawContours( blank, contours, i, color, 1, 8);
        img_preview(blank);
        double area=contourArea(contours[i]);
    }




    //img_preview(bw_img);


 	return 0; 
}

void bgr2BW(Mat bgrimg,Mat *BW_res)
{
    Mat channel[3];  
    Mat E(bgrimg.size(),CV_8UC1,Scalar(255));
    Mat BW(bgrimg.size(),CV_8UC1,Scalar(0));
    Mat tmp;
    // img_preview(BW);
    // img_preview(E);
    Mat ele = getStructuringElement(MORPH_RECT,Size(3,3));//get kernel

    split(bgrimg,channel); 
    for(int i=0;i<3;i++)
    {
    tmp=channel[i].clone();

    //img_preview(tmp);

    threshold(tmp, tmp, 25, 255.0, CV_THRESH_BINARY);

    //img_preview(tmp);

    erode(tmp,tmp,ele);

    bitwise_or(tmp,BW,BW);

    // img_preview(BW);

    tmp=channel[i].clone();


    Mat grad_x,grad_y,abs_grad_x,abs_grad_y;
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    //Scharr( tmp, grad_x, CV_8UC1, 1, 0);
    GaussianBlur( tmp, tmp, Size(5,5), 0, 0, BORDER_DEFAULT );

    Sobel( tmp, grad_x, CV_8UC1, 2, 0, 3);
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    //Scharr( tmp, grad_y, CV_8UC1, 0, 1);
    Sobel( tmp, grad_y, CV_8UC1, 0, 2, 3);
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, tmp);
    // img_preview(tmp);
    threshold(tmp, tmp, 25, 255.0, CV_THRESH_BINARY);
    
    bitwise_not(tmp,tmp);
    erode(tmp,tmp,ele);

    bitwise_and(E,tmp,E);
    // img_preview(tmp);
}
    bitwise_and(E,BW,BW);
    erode(BW,BW,ele);
    morphologyEx( BW, ele, MORPH_OPEN, ele );
    *BW_res=BW.clone();

}


void img_preview(Mat img)
{
    imshow("img_preview",img);
    waitKey(0);
}

int Median(double a[],int N)
{
    int i,j,max;
    int t;
    for(i=0;i<N-1;i++)
    {
        max=i;
        for(j=i+1;j<N;j++)
        if(a[j]>a[max]) max=j;
        t=a[i];a[i]=a[max];a[max]=t;
    }
    return a[(N-1)/2];
 }