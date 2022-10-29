#include <cmath>
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include <opencv4/opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <config/param.h>
#include <std_msgs/UInt8.h>
#include <tf/tf.h>
#include <yolov5/result.h>
#include "calibrator.h"
#include "preprocess.h"

using namespace cv;

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define PI 3.14159265
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !
float NMS_THRESH = 0.4;
float CONF_THRESH = 0.5;
#define BATCH_SIZE 1


static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
int yaw, pitch, roll;
static Logger gLogger;
ros::Publisher detectingPub;
ros::Publisher resultPub;
void imageCallback(const sensor_msgs::ImageConstPtr& msg);
void* buffers[2];
static float data[3 * INPUT_H * INPUT_W];
static float prob[ OUTPUT_SIZE];
uint8_t* img_host = nullptr;
uint8_t* img_device = nullptr;
int inputIndex;
int outputIndex;
int exchange_distance;
cudaStream_t stream;
IExecutionContext* context;

double cameraDistance = 450;
enum target
{
    volleyball = 1, basketball = 2, basket = 3, mark = 4, home = 5
};
target aim = (target)1;
bool if_show = true;
class result_deal
{
public:
    cv::Rect bbox;
    float conf;
    int class_id;
    int score;
    int distance;
    int center_x;
    int center_y;
    int yaw;
    //double pitch;
    //double roll;
    
    result_deal(cv::Rect re = cv::Rect(0, 0, 0, 0), int id = 100, int y = 0)
    {
        class_id = id;
        bbox = re;
        center_x = (int)(bbox.x + 0.5 * bbox.width);
        center_y = (int)(bbox.y + 0.5 * bbox.height);
        getDistance();
        getScore();
        yaw = y;
        //ROS_INFO("roll%.12lf, pitch%.12lf, yaw%.12lf", roll / PI * 180, pitch / PI * 180, yaw/ PI * 180);
   }
    void getDistance()
    {
        double k = 20;
        int size;
        if(class_id == 3)
        {
            if(bbox.x <= 5 || (bbox.x + bbox.width) >= 635)
            {
                size = bbox.height;
                k = 100;
            }
            else if(bbox.y <= 5 || (bbox.y + bbox.height) >= 507)
            {
                size = bbox.width;
                k = 20;
            }
            else
            {
                size = bbox.width;
                k = 20;
            }
        }
        else if(bbox.x <= 5 || (bbox.x + bbox.width) >= 635)
        {
            size = bbox.height;
        }
        else if(bbox.y <= 5 || (bbox.y + bbox.height) >= 507)
        {
            size = bbox.width;
        }
        else
        {
            size = bbox.width;
        }
        if(class_id == 0)k = 21.0084524;
        else if(class_id == 1)k = 24.6;
        else if(class_id == 2)distance = exchange_distance;
        else if(class_id == 4)k = 100;
        distance = sqrt(k * k * ((center_x - 320) * (center_x - 320) + (center_y - 256) * (center_y - 256) + cameraDistance * cameraDistance) / (size * size));
        if(class_id == 2)
        {
            distance = exchange_distance;
        }
        if(class_id == 3)
        {
            exchange_distance = distance;
        }
        if((bbox.x <= 5 || (bbox.x + bbox.width) >= 635) && (bbox.y <= 5 || (bbox.y + bbox.height) >= 507))
        {
            if(class_id == 0 && bbox.y >= 246)
            {
                distance = (-1.3174 * bbox.y + 360.36);
            }
            else if(class_id == 1 && bbox.y >= 220)
            {
                distance = (-1.3285 * bbox.y + 314.2);
            }
            else if(bbox.width > 600)
            {
                distance = 10;
            }
            if(distance < 0)
            {
                distance = 0;
            }
        }
        //ROS_INFO("%d, %d", bbox.x + bbox.width, bbox.y + bbox.height);
    }

    void getScore()
    {
        if(aim == (target)1)
        {
            if(class_id == 2 || class_id == 3 || class_id == 4 || class_id == 1)
            {
                score = 0;
            }
            else
            {
                score = 1000 - distance - 2 * (center_x - 320);
            }
        }
        if(aim == (target)2)
        {
            if(class_id == 0 || class_id == 2 ||class_id == 3 ||class_id == 4)
            {
                score = 0;
            }
            else
            {
                score = 1000 - distance - 2 * (center_x - 320);
            }
        }
        if(aim == (target)4)
        {
            if(class_id == 0 || class_id == 1 || class_id == 4)
            {
                score = 0;
            }
            else if(class_id == 3)
            {
                score = 999;
            }
            else
            {
                score = 1000;
            }
        }
        if(aim == (target)3)
        {

            if(class_id == 0 || class_id == 1 || class_id == 2 || class_id == 4)
            {
                score = 0;
            }
            else
            {
                score = 1000;
            }
        }
        if(aim == (target)5)
        {
            if(class_id == 0 || class_id == 1 || class_id == 2 ||class_id == 3)
            {
                score = 0;
            }
            else
            {
                score = 1000;
            }
        }
        if(score < 0)
        {
            score = 0;
        }
    }
   
};

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void paramCallback(const config::param::ConstPtr& msg)
{
    NMS_THRESH = msg->NMS_THRESH;
    CONF_THRESH = msg->CONF_THRESH;
    if_show = msg->if_show;
}
void modeCallback(const std_msgs::UInt8::ConstPtr& msg)
{
    aim = (target)msg->data;
    ROS_INFO("%d", aim);
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
void findSquares(Mat& img, std::vector<Yolo::Detection> br )
{
    Mat canny;
	//GaussianBlur(img, img, Size(3, 3), 5);
	Canny(img, canny, 160, 300, 3);
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
			//putText(img, "rectangle", center, 0, 1, Scalar(0, 255, 0), 1, 8);
            int dis1, dis2, dis3, dis4;
            double d1, d2;
            dis1 = (result.at<Vec2i>(0)[0] - result.at<Vec2i>(1)[0]) * (result.at<Vec2i>(0)[0] - result.at<Vec2i>(1)[0]) + (result.at<Vec2i>(0)[1] - result.at<Vec2i>(1)[1]) * (result.at<Vec2i>(0)[1] - result.at<Vec2i>(1)[1]);
            dis2 = (result.at<Vec2i>(1)[0] - result.at<Vec2i>(2)[0]) * (result.at<Vec2i>(1)[0] - result.at<Vec2i>(2)[0]) + (result.at<Vec2i>(1)[1] - result.at<Vec2i>(2)[1]) * (result.at<Vec2i>(1)[1] - result.at<Vec2i>(2)[1]);
            dis3 = (result.at<Vec2i>(2)[0] - result.at<Vec2i>(3)[0]) * (result.at<Vec2i>(2)[0] - result.at<Vec2i>(3)[0]) + (result.at<Vec2i>(2)[1] - result.at<Vec2i>(3)[1]) * (result.at<Vec2i>(2)[1] - result.at<Vec2i>(3)[1]);
            dis4 = (result.at<Vec2i>(3)[0] - result.at<Vec2i>(0)[0]) * (result.at<Vec2i>(3)[0] - result.at<Vec2i>(0)[0]) + (result.at<Vec2i>(3)[1] - result.at<Vec2i>(0)[1]) * (result.at<Vec2i>(3)[1] - result.at<Vec2i>(0)[1]);
            d1 = sqrt(dis1+dis3);
            d2 = sqrt(dis2+dis4);
            float a = d1 / (d1 + d2);
            if(a > 0.3 && a < 0.7)
            {
                Yolo::Detection yd;
                yd.bbox[0] = center.x;
                yd.bbox[1] = center.y;
                yd.bbox[2] = 10;
                yd.bbox[3] = 10;
                yd.conf = 1;
                yd.class_id = 6;
            }
	
            ROS_INFO("%f",a);
		}
	}
}


int main(int argc, char** argv)
{
    cudaSetDevice(DEVICE);
    ros::init(argc, argv, "yolov5_node");
    ros::start();
    ros::NodeHandle n;
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream = nullptr;
    size_t size = 0;
    std::string engine_name(PROJECT_PATH);
    engine_name += "/engine/yolov5s.engine";
    std::ifstream file(engine_name, std::ios::binary);
    if(file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    // prepare input data ---------------------------
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

    ROS_WARN("engine init done!");
    ros::Subscriber imageSub = n.subscribe("/MVCamera/image_raw", 1, &imageCallback);
    ros::Subscriber paramSub = n.subscribe("/param", 1, &paramCallback);
    ros::Subscriber modeSub = n.subscribe("/mode", 1, &modeCallback);
    detectingPub = n.advertise<sensor_msgs::Image>("/detectingResult", 1);
    resultPub = n.advertise<yolov5::result>("/target", 1);
    ros::spin();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    float* buffer_idx = (float*)buffers[inputIndex];/////////////////////////////16
    if (!img.empty())
    {
        size_t  size_image = img.cols * img.rows * 3;////////////////////7
        size_t  size_image_dst = INPUT_H * INPUT_W * 3;//////////////////////8
       	//copy data to pinned memory
        memcpy(img_host,img.data,size_image);//////////////////////////9
        //copy data to device memory
        CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));//////////////////10
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream); //////////////11      
        buffer_idx += size_image_dst;///////////////////////////12
    }
    // Run inference
    std::vector<result_deal> pass_result;
    std::vector<Yolo::Detection> batch_res;
    if(aim != 4)
    {
        doInference(*context, stream, (void**)buffers, prob, 1);////////////////////////////13
        nms(batch_res, &prob[0], CONF_THRESH, NMS_THRESH);
    }
    else
    {
        Mat img_gray;
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        findSquares(img_gray, batch_res);
	    imshow("gray", img_gray);
	    waitKey(30);
    }
    ros::param::get("/yaw", yaw);
    for (int b = 0; b < batch_res.size(); b++)
    {
        cv::Rect r = get_rect(img, batch_res[b].bbox);
        result_deal rd(r, batch_res[b].class_id, yaw);
        cv::putText(img, std::to_string(rd.score), cvPoint(r.x, r.y - 29), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        cv::putText(img, std::to_string(rd.distance), cvPoint(r.x, r.y - 43), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        if(pass_result.empty())
        {
            pass_result.push_back(rd);
        }
        else if(rd.score >= pass_result[0].score)
        {
            pass_result.insert(pass_result.begin(), rd);
        }
        else 
        {
            pass_result.push_back(rd);
        }
    }
    if(pass_result.empty() || pass_result[0].score == 0)
    {
        yolov5::result msg;
        msg.x = 0xFFFFFFFF;
        msg.y = 0xFFFFFFFF;
        msg.distance = 0xFFFFFFFF;
        msg.direction = 0xFFFFFFFF;
        resultPub.publish(msg);
    }
    else
    {
        yolov5::result msg;
        msg.x = pass_result[0].center_x;
        msg.y = pass_result[0].center_y;
        msg.distance = pass_result[0].distance;
        msg.direction = pass_result[0].yaw;
        resultPub.publish(msg);

    }
    pass_result.clear();
    if(if_show)
    {
        for (int b = 0; b < batch_res.size(); b++) 
        {
            //auto& res = batch_res[b];
            for (size_t j = 0; j < batch_res.size(); j++) 
            {
                cv::Rect r = get_rect(img, batch_res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 1);
                cv::putText(img, std::to_string((int)batch_res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
                cv::putText(img, std::to_string(batch_res[j].conf), cvPoint(r.x, r.y - 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
            }
        }
        sensor_msgs::ImagePtr detectingResult = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
        detectingPub.publish(detectingResult);
    }
}
