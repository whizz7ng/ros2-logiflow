#include "myagv_odometry/myAGV.h"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdint>

double linearX = 0.0;
double linearY = 0.0;
double angularZ = 0.0;

rclcpp::Time last_cmd_time;

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MyAGV>("myagv_odometry_node");

    if (!node->init()) {
        RCLCPP_ERROR(node->get_logger(), "myAGV initialized failed!");
        return 1;
    }

    RCLCPP_INFO(node->get_logger(), "myAGV initialized successful!");

    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.best_effort();

    last_cmd_time = node->now();

    auto sub = node->create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel",
        qos,
        [&node](const geometry_msgs::msg::Twist::SharedPtr msg)
        {
            static uint64_t cmd_rx_seq = 0;
            static double prev_x = 999.0;
            static double prev_y = 999.0;
            static double prev_w = 999.0;

            cmd_rx_seq++;

            linearX = msg->linear.x;
            linearY = msg->linear.y;
            angularZ = msg->angular.z;
            last_cmd_time = node->now();

            const bool changed =
                std::fabs(linearX - prev_x) > 1e-4 ||
                std::fabs(linearY - prev_y) > 1e-4 ||
                std::fabs(angularZ - prev_w) > 1e-4;

            if (changed) {
                RCLCPP_INFO(
                    node->get_logger(),
                    "[CMD_RX] seq=%llu t=%.3f vx=%.4f vy=%.4f wz=%.4f",
                    static_cast<unsigned long long>(cmd_rx_seq),
                    node->now().seconds(),
                    linearX,
                    linearY,
                    angularZ
                );

                prev_x = linearX;
                prev_y = linearY;
                prev_w = angularZ;
            } else {
                RCLCPP_INFO_THROTTLE(
                    node->get_logger(),
                    *node->get_clock(),
                    1000,
                    "[CMD_RX_KEEP] seq=%llu t=%.3f vx=%.4f vy=%.4f wz=%.4f",
                    static_cast<unsigned long long>(cmd_rx_seq),
                    node->now().seconds(),
                    linearX,
                    linearY,
                    angularZ
                );
            }
        }
    );

    rclcpp::Rate loop_rate(100);

    bool timeout_active = false;

    while (rclcpp::ok()) {
        rclcpp::spin_some(node);

        const double cmd_age = (node->now() - last_cmd_time).seconds();

        if (cmd_age > 0.3) {
            if (!timeout_active) {
                RCLCPP_WARN(
                    node->get_logger(),
                    "[CMD_TIMEOUT] no /cmd_vel for %.3f sec -> force zero command",
                    cmd_age
                );
                timeout_active = true;
            }

            linearX = 0.0;
            linearY = 0.0;
            angularZ = 0.0;
        } else {
            timeout_active = false;
        }

        node->execute(linearX, linearY, angularZ);

        loop_rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
