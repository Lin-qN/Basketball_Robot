#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <config/RC2021Config.h>
#include <config/param.h>
ros::Publisher param_pub;

void dynamicCallback(config::RC2021Config &config, uint32_t level);
 
int main(int argc, char **argv) 
{
    ros::init(argc, argv, "config_node");
    ros::NodeHandle n;

    param_pub = n.advertise<config::param>("/param", 10);
    dynamic_reconfigure::Server<config::RC2021Config> server;
    dynamic_reconfigure::Server<config::RC2021Config>::CallbackType f;
    f = boost::bind(&dynamicCallback, _1, _2);
    server.setCallback(f);
    ros::spin();
    return 0;
}
void dynamicCallback(config::RC2021Config &config, uint32_t level)
{
    config::param param_msg;

    param_msg.CONF_THRESH = 0.7;
    if(param_msg.CONF_THRESH != config.CONF_THRESH)
    {
        param_msg.CONF_THRESH = config.CONF_THRESH;
        param_pub.publish(param_msg);
    }

    param_msg.NMS_THRESH = 0.4;
    if(param_msg.NMS_THRESH != config.NMS_THRESH)
    {
        param_msg.NMS_THRESH = config.NMS_THRESH;
        param_pub.publish(param_msg);
    }

    param_msg.if_shot = 0;
    if(param_msg.if_shot != config.if_shot)
    {
        param_msg.if_shot = config.if_shot;
        param_pub.publish(param_msg);
    }

    param_msg.if_show = 1;
    if(param_msg.if_show != config.if_show)
    {
        param_msg.if_show = config.if_show;
        param_pub.publish(param_msg);
    }

    param_msg.k_x0 = 0;
    if(param_msg.k_x0 != config.k_x0)
    {
        param_msg.k_x0 = config.k_x0;
        param_pub.publish(param_msg);
    }

    param_msg.k_y0 = 0;
    if(param_msg.k_y0 != config.k_y0)
    {
        param_msg.k_y0 = config.k_y0;
        param_pub.publish(param_msg);
    }

    param_msg.k_z0 = 0;
    if(param_msg.k_z0 != config.k_z0)
    {
        param_msg.k_z0 = config.k_z0;
        param_pub.publish(param_msg);
    }

    param_msg.k_x1 = 255;
    if(param_msg.k_x1 != config.k_x1)
    {
        param_msg.k_x1 = config.k_x1;
        param_pub.publish(param_msg);
    }

    param_msg.k_y1 = 255;
    if(param_msg.k_y1 != config.k_y1)
    {
        param_msg.k_y1 = config.k_y1;
        param_pub.publish(param_msg);
    }

    param_msg.k_z1 = 255;
    if(param_msg.k_z1 != config.k_z1)
    {
        param_msg.k_z1 = config.k_z1;
        param_pub.publish(param_msg);
    }

}