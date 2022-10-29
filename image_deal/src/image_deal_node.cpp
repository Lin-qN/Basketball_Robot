#include <ros/ros.h>
#include <yolov5/result.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <config/param.h>
#include <vector>
using namespace cv;
char ad[100] = { 0 };
bool if_shot = false;
int hs = 0, ss = 0, vs = 0, hb = 255, sb = 255, vb = 255;
int hsv = 0, ssv = 0, vsv = 0, hbv = 255, sbv = 255, vbv = 255;
void imageDealCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if (if_shot)
    {
        cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        static int d = 10000;
        cv::imshow("shot", img);
        int k = cv::waitKey(30);
        if(k == 's' && !img.empty())
        {
            sprintf(ad, "/home/dovejh/project/RC/train/images/%d.jpg", d++);
            cv::imwrite(ad, img);
            ROS_INFO("Get %d pictures!", d);
        }
    }
}
void readVideo()
{
    cv::VideoCapture video("/home/dovejh/project/RC/video/3.mp4");
    if(video.isOpened())
    {
        ROS_WARN("Video load done!");
    }
    else
    {
        ROS_ERROR("Can't open video!");
    }
    while(1)
    {
        cv::Mat frame;
        video >> frame;
        if(frame.empty())
        {
            break;
        }
        cv::imshow("video", frame);
        cv::waitKey(1000 / video.get(cv::CAP_PROP_FPS));
    }
    cv::waitKey();
}
void paramCallback(const config::param::ConstPtr& msg)
{
    if_shot = msg->if_shot;
    hs = msg->k_x0;
    ss = msg->k_y0;
    vs = msg->k_z0;
}
void squarePerspectiveTransformation(Mat& img, Point lt, Point rt, Point rb, Point lb)
{

}
void drawapp(Mat result, Mat img2)
{
	for (int i = 0; i < result.rows; i++)
	{
		//最后一个坐标点与第一个坐标点连接
		if (i == result.rows - 1)
		{
			Vec2i point1 = result.at<Vec2i>(i);
			Vec2i point2 = result.at<Vec2i>(0);
			line(img2, point1, point2, Scalar(0, 0, 255), 2, 8, 0);
			break;
		}
		Vec2i point1 = result.at<Vec2i>(i);
		Vec2i point2 = result.at<Vec2i>(i + 1);
		line(img2, point1, point2, Scalar(0, 0, 255), 2, 8, 0);
	}
}
void findSquares(Mat& img)
{
    Mat canny;
	Canny(img, canny, 80, 160, 3, false);
	//膨胀运算
	Mat kernel = getStructuringElement(0, Size(3, 3));
	dilate(canny, canny, kernel);

	// 轮廓发现与绘制
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	findContours(canny, contours, hierarchy, 0, 2, Point());
    int t;
	//绘制多边形
	for (t = 0; t < contours.size(); t++)
	{
        if(contourArea(contours[t]) < 10000)continue;
		//用最小外接矩形求取轮廓中心
		RotatedRect rrect = minAreaRect(contours[t]);
		Point2f center = rrect.center;
		circle(img, center, 2, Scalar(0, 255, 0), 2, 8, 0);

		Mat result;
		approxPolyDP(contours[t], result, 4, true);  //多边形拟合
		drawapp(result, img);

		//判断形状和绘制轮廓
		if (result.rows == 4)
		{
			putText(img, "rectangle", center, 0, 1, Scalar(0, 255, 0), 1, 8);
            int dis1, dis2, dis3, dis4;
            double d1, d2;
            dis1 = (result.at<Vec2i>(0)[0] - result.at<Vec2i>(1)[0]) * (result.at<Vec2i>(0)[0] - result.at<Vec2i>(1)[0]) + (result.at<Vec2i>(0)[1] - result.at<Vec2i>(1)[1]) * (result.at<Vec2i>(0)[1] - result.at<Vec2i>(1)[1]);
            dis2 = (result.at<Vec2i>(1)[0] - result.at<Vec2i>(2)[0]) * (result.at<Vec2i>(1)[0] - result.at<Vec2i>(2)[0]) + (result.at<Vec2i>(1)[1] - result.at<Vec2i>(2)[1]) * (result.at<Vec2i>(1)[1] - result.at<Vec2i>(2)[1]);
            dis3 = (result.at<Vec2i>(2)[0] - result.at<Vec2i>(3)[0]) * (result.at<Vec2i>(2)[0] - result.at<Vec2i>(3)[0]) + (result.at<Vec2i>(2)[1] - result.at<Vec2i>(3)[1]) * (result.at<Vec2i>(2)[1] - result.at<Vec2i>(3)[1]);
            dis4 = (result.at<Vec2i>(3)[0] - result.at<Vec2i>(0)[0]) * (result.at<Vec2i>(3)[0] - result.at<Vec2i>(0)[0]) + (result.at<Vec2i>(3)[1] - result.at<Vec2i>(0)[1]) * (result.at<Vec2i>(3)[1] - result.at<Vec2i>(0)[1]);
            d1 = sqrt(dis1+dis3);
            d2 = sqrt(dis2+dis4);
            ROS_INFO("%lf %lf",d1, d2 );
		}
	}
}
static void callback1(int, void*)
{
    hs = hsv;
}
static void callback2(int, void*)
{
    ss = ssv;
}
static void callback3(int, void*)
{
    vs = vsv;
}
int main(int argc, char **argv)
{
    //system("color F0");
    //readVideo();
    ros::init(argc, argv, "image_deal_node");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("/MVCamera/image_raw", 1, &imageDealCallback);
    ros::Subscriber paramSub = n.subscribe("/param", 1, &paramCallback);
    //cv::Mat img = cv::imread("/home/dovejh/project/RC/train/images/1281.jpg"), img2;
    //cv::imshow("raw", img);
    //namedWindow("Trackbar");
    //createTrackbar("hs", "Trackbar", &hsv, 255, callback1);
    //createTrackbar("ss", "Trackbar", &ssv, 255, callback2);
    //createTrackbar("vs", "Trackbar", &vsv, 255, callback3);
    //img.convertTo(img, CV_32F, 1.0 / 255);
    //cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        //cv::threshold(imgs1[0], imgs1[0], hs, 255, cv::THRESH_BINARY);
        //cv::threshold(imgs1[1], imgs1[1], vs, 255, cv::THRESH_BINARY);
        //cv::threshold(imgs1[2], imgs1[2], ss, 255, cv::THRESH_BINARY);
        //cv::bitwise_and(imgs1[0], imgs1[1], img_bin);
        //cv::bitwise_and(img_bin, imgs1[2], img_bin);
        //std::vector<Vec2f> lines;
        //findSquares(img);
        //cv::imwrite("/home/dovejh/project/RC/train/images/9999.jpg", img);
        /*HoughLines(img_bin, lines,1, CV_PI / 180,  50, 0, 0);
        for(size_t i = 0; i < lines.size(); i++)
        {
            Vec4i l = lines[i];
            line(img_bin, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0 ,0 , 255), 3, 16);
        } */
        //cv::imshow("line", img_bin);
        //ROS_INFO("%d, %d, %d", hb, sb, vb);
        
        //ros::spinOnce();
       // while(1)
        //{
            //cv::imshow("bindary", img);
            //cv::waitKey(30);
        //}
    ros::spin();
}
