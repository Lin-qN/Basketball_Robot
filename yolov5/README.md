## 介绍
将yolov5s -v6.0与ROS相结合
从/MVCamera/image_raw话题订阅图像，并将识别结果发布到/detectingResult话题里
yolov5和tensorrtx的使用方式见https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5
## 依赖环境
CUDA tensorrt Python3 ROS1 Opencv4
## 部署
1. 根据https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5的教程生成engine文件
2. 将engine文件放入yolov5_with_ros/engine文件夹下
3. yolov5_with_ros/src/yolov5.cpp中根据你的需要修改输入图像和输出图像话题（https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5中需要修改的内容也需要一并修改）
## 特别注意
1. 注意yolov5和tensorrt的版本对应问题
2. engine和wts的转化问题可以查看https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5的FAQ部分
3. yolov5_with_ros下的参数一定要与tensorrtx下的参数保持一致，否则会导致识别不到的问题。（解决方案：将两部分修改一致后重新从wts文件生成engine文件）
4. 如果tensorrtx安装路径不是默认路径需要修改cmake
5. 创建者:刘建航 QQ：1358446393 Tel：18730297258 E-mail：1358446393@qq.com
