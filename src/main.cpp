
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
void crossPoint(Point p0,Point p1,double angle0,double angle1,Point* crossPoint);
void drawsmallCross(Mat img,Point p,Scalar color);
int pointCluster(vector<Point> raw,vector<Point>* center,double limit);
void checkPixelColor(Mat raw_copy,Mat* HSV,Point po);
void areaOffset(vector<Point> src,vector<Point>* des,Point src_cen,Point des_cen);
void drawPoints(Mat img,vector<Point> po,const Scalar & color);
void putCentertext(Mat img,const char * str,Point po,int font,int size,Scalar color);

static void onmouse(int event,int x,int y,int f,void * param);

int main(int argc, char** argv)
{
	Mat raw_img = imread(imgname);
    Mat bw_img;
    Mat HSVchannel[3];
    namedWindow("raw",CV_WINDOW_NORMAL);
    namedWindow("img_preview",CV_WINDOW_NORMAL);
    cv::setMouseCallback("img_preview",onmouse,reinterpret_cast<void*> (&HSVchannel[0]));  
    imshow("raw",raw_img);
    bgr2BW(raw_img,&bw_img);
    int i,j;
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
    vector<double>  faceArea;
    //vector<int>  faceFlag;
    ULq.reserve(numFace);
    URq.reserve(numFace);
    DLq.reserve(numFace);
    DRq.reserve(numFace);
    faceCenter.reserve(numFace);
    faceArea.reserve(numFace);
    //faceFlag.reserve(numFace);
    int faceFlag[numFace]={0};//0-init 1-left 2-right 3-top


    for(i=1;i<numFace;i++)
    {
        areaSizeQuene[i]=contourArea(contours[i]);
    }




    for(i=0; i<contours.size(); i++ )
    {
        Scalar color=255;
        drawContours( blank, contours, i, color, -1, 8);
        //img_preview(blank);
        double area=contourArea(contours[i]);
        faceArea[i]=area;
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
    //masked img
    Mat img_masked;
    raw_img.copyTo(img_masked,blank);
    img_preview(img_masked);    






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

    Mat hsv_channel[3];
   img_preview(raw_copy); 




   for(i=0;i<numFace;i++)
   {
    drawContours( raw_copy, contours, i, Scalar(255,255,255), 2, 8);
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
    
    //find cross points of all this lines
    double Upper_corre=1.1;
    vector<Point> Lcross;
    vector<Point> Rcross;
    vector<Point> Ucross;
    double areaL=0;
    double areaR=0;
    double areaU=0;


    for(i=0;i<numFace;i++)
   {

    if(faceFlag[i]==1)//left
    {
        //cout<<"UL UR DL DR "<<ULq[i]<<" "<<URq[i]<<"  "<<DLq[i]<<"  "<<DRq[i]<<endl;
        //drawXline(raw_copy,faceCenter[i],MedangleXL,Scalar(255,0,0));
        //drawXline(raw_copy,faceCenter[i],MedangleYL,Scalar(0,255,0));
        for(j=0;j<numFace;j++)
       {
            if((faceFlag[j]==1)&&(i!=j))//left
            {
                Point cross;
                crossPoint(faceCenter[i],faceCenter[j],MedangleXL,MedangleYL,&cross);
                Lcross.push_back(cross);
                areaL+=faceArea[i];
                //drawsmallCross(raw_copy,cross,Scalar(255,0,0));
            }
       }
    }
    
    if(faceFlag[i]==2)//right
    {
        //drawXline(raw_copy,faceCenter[i],MedangleXR,Scalar(0,0,0));
       //drawXline(raw_copy,faceCenter[i],MedangleYR,Scalar(255,255,255));
        for(j=0;j<numFace;j++)
       {
        if((faceFlag[j]==2)&&(i!=j))//left
        {
            Point cross;
            crossPoint(faceCenter[i],faceCenter[j],MedangleXR,MedangleYR,&cross);
            Rcross.push_back(cross);
            areaR+=faceArea[i];
            //drawsmallCross(raw_copy,cross,Scalar(255,0,0));
        }
        }
    }
    if(faceFlag[i]==3)//upper
    {
        drawXline(raw_copy,faceCenter[i],MedangleXL,Scalar(255,0,0));
        drawXline(raw_copy,faceCenter[i],MedangleXR*Upper_corre,Scalar(0,255,0));
        for(j=0;j<numFace;j++)
       {
        if((faceFlag[j]==3)&&(i!=j))//left
        {
            Point cross;
            crossPoint(faceCenter[i],faceCenter[j],MedangleXL,MedangleXR*Upper_corre,&cross);
            Ucross.push_back(cross);
            areaU+=faceArea[i];
            //drawsmallCross(raw_copy,cross,Scalar(0,255,0));
        }
        }
    }
   }
   areaL=areaL/cntL;
   areaR=areaR/cntR;
   areaU=areaU/cntU;
   double edgeL=sqrt(areaL);
   double edgeR=sqrt(areaR);
   double edgeU=sqrt(areaU);

   //delete some points which is close to central points in the same face

   double errLim=0.5;
   for(vector<Point>::iterator it=Lcross.begin();it!=Lcross.end();)
   {
    int delFlag=0;
    for(i=1;i<numFace;i++)
    {
        if(faceFlag[i]==1)
        {
            if(dist((*it),faceCenter[i])<edgeL*errLim)
            {
                delFlag=1;
                break;
            }
        }   
    }
    if(delFlag)
    {it=Lcross.erase(it);}
    else
    {it++;}
   }

    for(vector<Point>::iterator it=Rcross.begin();it!=Rcross.end();)
   {
    int delFlag=0;
    for(i=1;i<numFace;i++)
    {
        if(faceFlag[i]==2)
        {
            if(dist((*it),faceCenter[i])<edgeR*errLim)
            {
                delFlag=1;
                break;
            }
        }   
    }
    if(delFlag)
    {it=Rcross.erase(it);}
    else
    {it++;}
   }

    for(vector<Point>::iterator it=Ucross.begin();it!=Ucross.end();)
   {
    int delFlag=0;
    for(i=1;i<numFace;i++)
    {
        if(faceFlag[i]==3)
        {
            if(dist((*it),faceCenter[i])<edgeU*errLim)
            {
                delFlag=1;
                break;
            }
        }   
    }
    if(delFlag)
    {it=Ucross.erase(it);}
    else
    {it++;}
   }

   cout<<"Lcross"<<Lcross.size()<<endl;
   cout<<"Rcross"<<Rcross.size()<<endl;
   cout<<"Ucross"<<Ucross.size()<<endl;
   

   //point Clouster based seed point
   vector<Point> extraL;
   vector<Point> extraR;
   vector<Point> extraU;
   cout<<"LNUM: "<<pointCluster(Lcross,&extraL,edgeL*errLim)<<endl;
   cout<<"RNUM: "<<pointCluster(Rcross,&extraR,edgeR*errLim)<<endl;
   cout<<"UNUM: "<<pointCluster(Ucross,&extraU,edgeU*errLim)<<endl;

   //TODO draw miss retc here which is similiar to other rect


   //deal with color things
   Mat raw_color=raw_img.clone();
   Mat hsv_img;//=bgr2hsv(raw_color);

   cvtColor(raw_color,hsv_img,CV_BGR2HSV);
   split(hsv_img,HSVchannel);
   // img_preview(HSVchannel[0]);
   // img_preview(HSVchannel[1]);
   // img_preview(HSVchannel[2]);
    
   int font=FONT_HERSHEY_PLAIN;
   //scan all the poins and give color label
   for(i=0;i<numFace;i++)
   {
        //putText(raw_copy,"H",faceCenter[i],font,2,Scalar(0,0,0));

    //color porblem
        if((HSVchannel[1].at<uchar>(faceCenter[i]))<50)
        {
        putCentertext(raw_copy,"W",faceCenter[i],font,3,Scalar(0,0,0));
        continue;
        }
        int Hval=HSVchannel[0].at<uchar>(faceCenter[i]);
        if(Hval<10)
        {
        putCentertext(raw_copy,"R",faceCenter[i],font,3,Scalar(0,0,0));
        }
        if(Hval>9&&Hval<22)
        {
        putCentertext(raw_copy,"O",faceCenter[i],font,3,Scalar(0,0,0));
        }
        if(Hval>21&&Hval<35)
        {
        putCentertext(raw_copy,"Y",faceCenter[i],font,3,Scalar(0,0,0));
        }
        if(Hval>50&&Hval<90)
        {
        putCentertext(raw_copy,"G",faceCenter[i],font,3,Scalar(0,0,0));
        }
        if(Hval>95&&Hval<120)
        {
        putCentertext(raw_copy,"B",faceCenter[i],font,3,Scalar(0,0,0));
        }
        
   }

   if(extraL.size()!=0)
   {
        for(i=0;i<extraL.size();i++)
        {
            checkPixelColor(raw_copy,HSVchannel,extraL[i]);
        }
   }
   if(extraR.size()!=0)
   {
        for(i=0;i<extraR.size();i++)
        {
            checkPixelColor(raw_copy,HSVchannel,extraR[i]);
        }
   }   
   if(extraU.size()!=0)
   {
        for(i=0;i<extraU.size();i++)
        {
            checkPixelColor(raw_copy,HSVchannel,extraU[i]);
        }
   }
   //find fit area for each direction face(actually smallest center)


   Point LfitCen;
   Point RfitCen;
   Point UfitCen;

   int Lfit,Rfit,Ufit;
   int smallestSizeL,smallestSizeR,smallestSizeU;
   smallestSizeL=100000;
   smallestSizeR=100000;
   smallestSizeU=100000;


   for(i=0;i<numFace;i++)
   {
        if(faceFlag[i]==1)
        {
            if(areaSizeQuene[i]<smallestSizeL)
            {
                Lfit=i;
                smallestSizeL=areaSizeQuene[i];
            }
        }
        if(faceFlag[i]==2)
        {
            if(areaSizeQuene[i]<smallestSizeR)
            {
                Rfit=i;
                smallestSizeR=areaSizeQuene[i];
            }
        }
        if(faceFlag[i]==3)
        {
            if(areaSizeQuene[i]<smallestSizeU)
            {
                Ufit=i;
                smallestSizeU=areaSizeQuene[i];
                //drawsmallCross(raw_copy,faceCenter[i],Scalar(0,0,255));
                cout<<"area:  "<<areaSizeQuene[i]<<endl;
            }
        } 
   }

    cout<<"fitnum: "<<Lfit<<" "<<Rfit<<" "<<Ufit<<endl;

    vector<vector<Point> > LfitArea;
    vector<vector<Point> > RfitArea;
    vector<vector<Point> > UfitArea;

   LfitArea.reserve(1);
   RfitArea.reserve(1);
   UfitArea.reserve(1);




    // (*(&RfitArea[0])).push_back(faceCenter[1]);
     //(*(&RfitArea[0])).push_back(faceCenter[2]);
    // cout<<RfitArea[0]<<endl;
    //cout<<contours[Rfit]<<endl;


   vector<Point> temp;

   //    for(i=0;i<extraR.size();i++)
   // {
   //  areaOffset(contours[Rfit],&temp,faceCenter[Rfit],extraR[0]);
   //  RfitArea.push_back(temp);
   //  drawContours(raw_copy, RfitArea,0, Scalar(255,255,255), 2, 8);
   // }
   //extraR.size()
   for(i=0;i<extraR.size();i++)
   {
    //RfitArea.clear();
    areaOffset(contours[Rfit],&temp,faceCenter[Rfit],extraR[i]);
    
    drawPoints(raw_copy,temp,Scalar(255,255,255));
    //RfitArea.push_back(temp);

    //drawContours(raw_copy, RfitArea,i, Scalar(255,255,255), 1,8,noArray(),1);

    //drawContours(raw_copy, RfitArea,0, Scalar(255,255,255), 2, 4);
   }
      for(i=0;i<extraL.size();i++)
   {
    //RfitArea.clear();
    areaOffset(contours[Lfit],&temp,faceCenter[Lfit],extraL[i]);
    
    //LfitArea.push_back(temp);
drawPoints(raw_copy,temp,Scalar(255,255,255));
    //drawContours(raw_copy, LfitArea,i, Scalar(255,255,255), 1, 8,noArray(),1);

    //drawContours(raw_copy, RfitArea,0, Scalar(255,255,255), 2, 4);

   }
      for(i=0;i<extraU.size();i++)
   {
    //RfitArea.clear();
    areaOffset(contours[Ufit],&temp,faceCenter[Ufit],extraU[i]);
    
   // UfitArea.push_back(temp);
drawPoints(raw_copy,temp,Scalar(255,255,255));
    //drawContours(raw_copy, UfitArea,i, Scalar(255,255,255), 1, 8,noArray(),1);

    //drawContours(raw_copy, RfitArea,0, Scalar(255,255,255), 2, 4);
    img_preview(raw_copy);
   }

    img_preview(raw_copy);












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


void drawsmallCross(Mat img,Point p,Scalar color)
{
    Point h(0,10);
    Point v(10,0);
    line(img,p-h,p+h,color,1,8,0);
    line(img,p-v,p+v,color,1,8,0);
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

void crossPoint(Point p0,Point p1,double angle0,double angle1,Point* crossPoint)
{
    (*crossPoint).y=(p0.y*angle0-p1.y*angle1-p0.x+p1.x)/(angle0-angle1);
    (*crossPoint).x=((double)((*crossPoint).y-p0.y)*angle0+p0.x);
}

int pointCluster(vector<Point> raw,vector<Point>* center,double limit)
{

    int label=0;//lable number
    int number=raw.size();

    if(number==0)// check if vector is empty
    {
        cout<<"empty point set"<<endl;
        return 0;
    }



    int initFlag=0;
    int finishFlag=0;
    int Flag[number]={0};//0-init 1-n label
    Point seed;


    while(finishFlag==0)
    {


        for(int i=0;i<number;i++)
        {
            if(initFlag==0)
            {
                if(Flag[i]==0)
                {
                initFlag=1;
                label++;
                seed=raw[i];
                Flag[i]=label;

                }
            }
            else
            {
                if(Flag[i]==0)
                {
                    if(dist(seed,raw[i])<limit)
                    {
                        Flag[i]=label;
                    }
                }
            }
            if(i==(number-1))
            {
                if(initFlag==0)
                {
                    finishFlag=1;
                }
                else
                {
                    initFlag=0;
                }
            }
        }
    }
    //calc center
    Point Pointsum[label+1];
    int cnt[label+1]={0};
    for(int i=0;i<number;i++)
    {
        Pointsum[Flag[i]]+=raw[i];
        cnt[Flag[i]]++;
    }
    for(int i=1;i<label+1;i++)
    {
        Pointsum[i].x=Pointsum[i].x/cnt[i];
        Pointsum[i].y=Pointsum[i].y/cnt[i];
        (*center).push_back(Pointsum[i]);
    }
    return label;
}

static void onmouse(int event,int x,int y,int f,void * param)
{
         Mat *im = reinterpret_cast<Mat*>(param);  
        
        if(event==CV_EVENT_LBUTTONDOWN)
        {
            std::cout<<"at("<<x<<","<<y<<")value is:"  
                <<static_cast<int>(im->at<uchar>(cv::Point(x,y)))<<std::endl;  
        }
}

void putCentertext(Mat img,const char * str,Point po,int font,int size,Scalar color)
{
    int baseLine;
    Size text_size=getTextSize(str, font, size, 1,&baseLine);
    Point offset;
    offset.x=po.x-text_size.width/2;
    offset.y=po.y+text_size.height/2;
    putText(img,str,offset,font,size-1,color,2);
}



void checkPixelColor(Mat raw_copy,Mat* HSV,Point po)
{
        int font=FONT_HERSHEY_PLAIN;
        if((HSV[1].at<uchar>(po))<50)
        {
        putCentertext(raw_copy,"W",po,font,3,Scalar(0,0,0));
        return;
        }
        int Hval=HSV[0].at<uchar>(po);
        if(Hval<10)
        {
        putCentertext(raw_copy,"R",po,font,3,Scalar(0,0,0));
        }
        if(Hval>9&&Hval<22)
        {
        putCentertext(raw_copy,"O",po,font,3,Scalar(0,0,0));
        }
        if(Hval>21&&Hval<35)
        {
        putCentertext(raw_copy,"Y",po,font,3,Scalar(0,0,0));
        }
        if(Hval>50&&Hval<90)
        {
        putCentertext(raw_copy,"G",po,font,3,Scalar(0,0,0));
        }
        if(Hval>95&&Hval<120)
        {
        putCentertext(raw_copy,"B",po,font,3,Scalar(0,0,0));
        }
}

void areaOffset(vector<Point> src,vector<Point>* des,Point src_cen,Point des_cen)
{
    Point offset;
    offset = des_cen - src_cen;
    //(*des).clear();



    for(int i=0;i<src.size();i++)
    {
        (*des).push_back(src[i]+offset);
    }
   
}
//TODO use average image replace specific image
void getAverageImg(vector<vector<Point> > src,vector<Point>* des)
{


}

void drawPoints(Mat img,vector<Point> po,const Scalar & color)
{
    for(int i=0;i<po.size();i++)
    {
        circle(img,po[i],0,color,2,8);
    }

}

