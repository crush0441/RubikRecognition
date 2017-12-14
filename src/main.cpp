
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
void CornerPoint(vector<Point> contour,Point* ULp,Point* URp,Point* DLp,Point *DRp);
void drawCross(Mat img,Point p,Scalar color);
double dist(Point p1,Point p2);
void drawXline(Mat img,Point po,double angle,Scalar color);
void drawYline(Mat img,Point po,double angle,Scalar color);
int main(int argc, char** argv)
{
	Mat raw_img = imread(imgname);
    Mat bw_img;
    namedWindow("raw",CV_WINDOW_NORMAL);
    namedWindow("img_preview",CV_WINDOW_NORMAL);
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
    
    //cout<<"medSize :"<<medSize<<endl;

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

    //cout<<contours.size()<<endl;
    // find corner point to divide faces
    Mat raw_copy=raw_img.clone();
    Point ULp,URp,DLp,DRp;

    int numFace = contours.size();
    vector<Point>  ULq,URq,DLq,DRq;
    vector<Point>  faceCenter;
    //vector<int>  faceFlag;
    ULq.reserve(numFace);
    URq.reserve(numFace);
    DLq.reserve(numFace);
    DRq.reserve(numFace);
    faceCenter.reserve(numFace);
    //faceFlag.reserve(numFace);
    int faceFlag[numFace]={0};//0-init 1-left 2-right 3-top

    for(i=0; i<contours.size(); i++ )
    {
        Scalar color=255;
        drawContours( blank, contours, i, color, 1, 8);
        
        double area=contourArea(contours[i]);
    Moments m;
    m=moments(contours[i],true);
    Point center;
    center.x=m.m10/m.m00;
    center.y=m.m01/m.m00;
    // drawCross(raw_copy,center,Scalar(125,0,0));
    faceCenter[i]=center;

    CornerPoint(contours[i],&ULp,&URp,&DLp,&DRp);
    // drawCross(raw_copy,ULp,Scalar(0,0,255));
    // drawCross(raw_copy,URp,Scalar(0,255,0));
    // drawCross(raw_copy,DLp,Scalar(255,0,0));
    //  drawCross(raw_copy,DRp,Scalar(255,255,255));




    ULq[i]=ULp;
    URq[i]=URp;
    DLq[i]=DLp;
    DRq[i]=DRp;

        //img_preview(blank);

    //find upper face
    double edgelen=sqrt(area);
    //cout<<"dist :"<<dist(ULp,DLp)<<endl;
    //cout<<"edge :"<<edgelen<<endl;
    if((dist(ULp,DLp)<edgelen/2)&&(dist(URp,DRp)<edgelen/2))
    {
        faceFlag[i]=3;
        //drawCross(raw_copy,center,Scalar(255,0,0));
    }


    //line have a big K is a vel line
    double dx1,dx2;
    if(abs(ULp.x-DLp.x)<0.00001)
    {dx1=3;}
    else
    {dx1=abs(ULp.y-DLp.y)/abs(ULp.x-DLp.x);}

    if(abs(URp.x-DRp.x)<0.00001)
    {dx2=3;}
    else
    {dx2=abs(URp.y-DRp.y)/abs(URp.x-DRp.x);}

    if((dx1>2.5)&&(dx2>2.5)&&faceFlag[i]==0)
    {
        if((URp.y>ULp.y)&&(DRp.y>DLp.y))
        {
            faceFlag[i]=1;//left
            //drawCross(raw_copy,center,Scalar(0,0,255));
        }
        if((URp.y<ULp.y)&&(DRp.y<DLp.y))
        {
            faceFlag[i]=2;//right
            //drawCross(raw_copy,center,Scalar(0,255,0));
        }
    }
      //img_preview(raw_copy); 
    }
    

    //find central point
    vector<double> XL;
    vector<double> YL;
    vector<double> XR;
    vector<double> YR;
    vector<double> XU;
    vector<double> YU;
    Point Lcen;
    Point Rcen;
    Point Ucen;
    int cntL,cntR,cntU;
    cntL=0;
    cntR=0;
    cntU=0;
        //cout<<"coufaceFlag "<<faceFlag.size()<<endl;
    for(i=0; i<numFace; i++ )
    { 

        if(faceFlag[i]==1)
        {  
        // cout<<"numFace "<<i<<endl;
            cntL++;
            XL.push_back(faceCenter[i].x);
            YL.push_back(faceCenter[i].y);
        }
        if(faceFlag[i]==2)
        {
        // cout<<"numFace "<<i<<endl;
            cntR++;
            XR.push_back(faceCenter[i].x);
            YR.push_back(faceCenter[i].y);
        }
        if(faceFlag[i]==3)
        {
            cntU++;
        // cout<<"numFace "<<i<<endl;
            // XU.push_back(faceCenter[i].x);
            // YU.push_back(faceCenter[i].y);
        }
    }
    //cout<<"XL: "<<XL.size()<<endl;
    ///cout<<Median(XL,XL.size());
    //find median value
    sort(XL.begin(),XL.end());
    sort(YL.begin(),YL.end());
    sort(XR.begin(),XR.end());
    sort(YR.begin(),YR.end());

    Lcen.x=XL[XL.size()/2];
    Lcen.y=YL[YL.size()/2];
    Rcen.x=XR[XR.size()/2];
    Rcen.y=YR[YR.size()/2];

   // drawCross(raw_copy,Lcen,Scalar(0,0,255));
   // drawCross(raw_copy,Rcen,Scalar(255,0,0));
   // drawCross(raw_copy,Ucen,Scalar(0,255,0));

   //find angle mayBe vel dx/dy
   vector<double> angleXL;
   vector<double> angleYL;
   vector<double> angleXR;
   vector<double> angleYR; 
   double MedangleXL,MedangleYL,MedangleXR,MedangleYR;


   img_preview(raw_copy); 


   for(i=0;i<numFace;i++)
   {
    drawContours( raw_copy, contours, i, Scalar(255,255,255), 1, 8);
    // /cout<<"time "<<i<<endl;
    if(faceFlag[i]==1)//left
    {
        // cout<<"UL UR DL DR "<<ULq[i]<<" "<<URq[i]<<"  "<<DLq[i]<<"  "<<DRq[i]<<endl;
        // drawXline(raw_copy,DLq[i],(double(DRq[i].x-DLq[i].x))/(DRq[i].y-DLq[i].y),Scalar(255,0,0));
        // //drawXline(raw_copy,ULq[i],(double(URq[i].x-ULq[i].x))/(URq[i].y-ULq[i].y),Scalar(0,255,0));
        // drawXline(raw_copy,DRq[i],(double(DRq[i].x-DLq[i].x))/(DRq[i].y-DLq[i].y),Scalar(0,0,255));
        // drawXline(raw_copy,URq[i],(double(DRq[i].x-DLq[i].x))/(DRq[i].y-DLq[i].y),Scalar(0,255,0));

        angleXL.push_back((double(DRq[i].x-DLq[i].x))/(DRq[i].y-DLq[i].y));
        angleXL.push_back((double(URq[i].x-ULq[i].x))/(URq[i].y-ULq[i].y));
       // img_preview(raw_copy);
        
        //cout<<(double(DRq[i].x-DLq[i].x)/(DRq[i].y-DLq[i].y))<<endl;

        angleYL.push_back((double(DLq[i].x-ULq[i].x))/(DLq[i].y-ULq[i].y));
        angleYL.push_back((double(DRq[i].x-URq[i].x))/(DRq[i].y-URq[i].y));
    }
    
    if(faceFlag[i]==2)//right
    {


        angleXR.push_back(double(DRq[i].x-DLq[i].x)/(DRq[i].y-DLq[i].y));
        angleXR.push_back(double(URq[i].x-ULq[i].x)/(URq[i].y-ULq[i].y));

    // cout<<(double(DRq[i].x-DLq[i].x)/(DRq[i].y-DLq[i].y))<<endl;
    // cout<<(double(URq[i].x-ULq[i].x)/(URq[i].y-ULq[i].y))<<endl;

        angleYR.push_back(double(DLq[i].x-ULq[i].x)/(DLq[i].y-ULq[i].y));
  
        angleYR.push_back(double(DRq[i].x-URq[i].x)/(DRq[i].y-URq[i].y));

        
    }
   }

    sort(angleXL.begin(),angleXL.end());
    sort(angleYL.begin(),angleYL.end());
    sort(angleXR.begin(),angleXR.end());
    sort(angleYR.begin(),angleYR.end());

    double vel_corr=0.;


    MedangleXL=angleXL[angleXL.size()/2];
    MedangleYL=angleYL[angleYL.size()/2];
    MedangleXR=angleXR[angleXR.size()/2];
    MedangleYR=vel_corr*angleYR[angleYR.size()/2];

    // cout<<"MedangleXL: "<<MedangleXL<<endl;
    // cout<<"MedangleYL: "<<MedangleYL<<endl;
    // cout<<"MedangleXR: "<<MedangleXR<<endl;
    // cout<<"MedangleYR: "<<angleYR.size()/2<<endl;

    //draw many lines
    //for upper face have correction C
    //TODO adjust C with area size
    double Upper_corre=1.1;
    for(i=0;i<numFace;i++)
   {

    if(faceFlag[i]==1)//left
    {
        //cout<<"UL UR DL DR "<<ULq[i]<<" "<<URq[i]<<"  "<<DLq[i]<<"  "<<DRq[i]<<endl;
        //drawXline(raw_copy,faceCenter[i],MedangleXL,Scalar(255,0,0));
        //drawXline(raw_copy,faceCenter[i],MedangleYL,Scalar(0,255,0));
    }
    
    if(faceFlag[i]==2)//right
    {
        //drawXline(raw_copy,faceCenter[i],MedangleXR,Scalar(0,0,0));
       //drawXline(raw_copy,faceCenter[i],MedangleYR,Scalar(255,255,255));
    }
    if(faceFlag[i]==3)//upper
    {
        drawXline(raw_copy,faceCenter[i],MedangleXL,Scalar(255,0,0));
        drawXline(raw_copy,faceCenter[i],MedangleXR*Upper_corre,Scalar(0,255,0));
    }
   }
    
    

    img_preview(raw_copy); 



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

void CornerPoint(vector<Point> contour,Point* ULp,Point* URp,Point* DLp,Point *DRp)
 {
    double UL,UR,DL,DR;
    double ULx,URx,DLx,DRx;

    UL=100000;
    UR=-100000;
    DL=100000;
    DR=-100000;
    
    ULx=100000;
    URx=-100000;
    DLx=100000;
    DRx=-100000;
    for(vector<Point>::iterator it=contour.begin();it!=contour.end();it++)
    {
        if(((*it).x+(*it).y<UL)||(((*it).x+(*it).y==UL)&&(*it).x<ULx))
        {
            (*ULp).x=(*it).x;
            (*ULp).y=(*it).y;
            ULx=(*it).x;
            UL=(*it).x+(*it).y;
        }
        if((*it).x-(*it).y>UR||(((*it).x-(*it).y==UR)&&(*it).x>URx))
        {
            //cout<<"DLp: DL"<<(*it).x-(*it).y<<"  "<<DL<<endl;
            (*URp).x=(*it).x;
            (*URp).y=(*it).y;
            URx=(*it).x;
            UR=(*it).x-(*it).y;
        }
        if((*it).x-(*it).y<DL||(((*it).x+(*it).y==DL)&&(*it).x<DLx))
        {
            (*DLp).x=(*it).x;
            (*DLp).y=(*it).y;
            DLx=(*it).x;
            DL=(*it).x-(*it).y;
        }
        if((*it).x+(*it).y>DR||(((*it).x-(*it).y==DR)&&(*it).x>DRx))
        {
            (*DRp).x=(*it).x;
            (*DRp).y=(*it).y;
            DRx==(*it).x;
            DR=(*it).x+(*it).y;
        }

    }
    // cout<<(*ULp)<<endl;
    // cout<<(*URp)<<endl;
    // cout<<(*DLp)<<endl;
    // cout<<(*DRp)<<endl;
 }

void drawCross(Mat img,Point p,Scalar color)
{
    Point h(0,20);
    Point v(20,0);
    line(img,p-h,p+h,color,2,8,0);
    line(img,p-v,p+v,color,2,8,0);
}

double dist(Point p1,Point p2)
{
    double res;
    Point p=p1-p2;
    res=sqrt(p.x*p.x+p.y*p.y);
    return res;
}
void drawXline(Mat img,Point po,double angle,Scalar color)
{
    //angel= dx/dy
    double len=200;
    //Point2d le(angle,1);
    Point l;
    Point r;
    l.x=po.x-angle*len;
    l.y=po.y-len;
    r.x=po.x+angle*len;
    r.y=po.y+len;
    line(img,l,r,color,2,8,0);
}

