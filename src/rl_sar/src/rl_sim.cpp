/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <cstdint>
#include <set>

namespace
{
constexpr float DEPTH_FX_PX = 106.0f * 11.041f / 20.955f;
constexpr float DEPTH_FY_PX = 60.0f * 11.041f / 12.240f;
constexpr float DEPTH_CX_PX = (106.0f - 1.0f) * 0.5f;
constexpr float DEPTH_CY_PX = (60.0f - 1.0f) * 0.5f;
constexpr float DEPTH_DEBUG_CLIP_MAX = 2.0f;

static float cubic_interp(float p0, float p1, float p2, float p3, float t)
{
    return p1 + 0.5f * t * (p2 - p0 + t * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + t * (3.0f * (p1 - p2) + p3 - p0)));
}

static void bicubic_resize(const float* src, int hs, int ws, float* dst, int hd, int wd)
{
    const float sy = static_cast<float>(hs) / static_cast<float>(hd);
    const float sx = static_cast<float>(ws) / static_cast<float>(wd);
    for (int y = 0; y < hd; ++y)
    {
        const float fy = (static_cast<float>(y) + 0.5f) * sy - 0.5f;
        const int y1 = static_cast<int>(std::floor(fy));
        const float ty = fy - static_cast<float>(y1);
        for (int x = 0; x < wd; ++x)
        {
            const float fx = (static_cast<float>(x) + 0.5f) * sx - 0.5f;
            const int x1 = static_cast<int>(std::floor(fx));
            const float tx = fx - static_cast<float>(x1);

            float col[4];
            for (int m = -1; m <= 2; ++m)
            {
                const int yy = std::clamp(y1 + m, 0, hs - 1);
                const float p0 = src[yy * ws + std::clamp(x1 - 1, 0, ws - 1)];
                const float p1 = src[yy * ws + std::clamp(x1, 0, ws - 1)];
                const float p2 = src[yy * ws + std::clamp(x1 + 1, 0, ws - 1)];
                const float p3 = src[yy * ws + std::clamp(x1 + 2, 0, ws - 1)];
                col[m + 1] = cubic_interp(p0, p1, p2, p3, tx);
            }
            dst[y * wd + x] = cubic_interp(col[0], col[1], col[2], col[3], ty);
        }
    }
}

static float sanitize_depth(float depth)
{
    if (!std::isfinite(depth))
    {
        return DEPTH_DEBUG_CLIP_MAX;
    }
    return depth;
}

static bool all_finite(const std::vector<float>& values)
{
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}

static bool env_flag_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (!value) return false;
    const std::string text(value);
    return !(text.empty() || text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF");
}

static float finite_or_zero(float value)
{
    return std::isfinite(value) ? value : 0.0f;
}

static float image_plane_depth_to_ray_depth(float z_depth, int u, int v)
{
    const float x = (static_cast<float>(u) - DEPTH_CX_PX) / DEPTH_FX_PX;
    const float y = (static_cast<float>(v) - DEPTH_CY_PX) / DEPTH_FY_PX;
    return z_depth * std::sqrt(1.0f + x * x + y * y);
}

static float parkour_command_x(float raw_x)
{
    if (!std::isfinite(raw_x) || std::abs(raw_x) < 1.0e-4f)
    {
        return 0.5f;
    }
    return std::clamp(raw_x, 0.3f, 0.8f);
}

static std::vector<float> build_parkour_joint_test_action(int step_count, int num_dofs)
{
    std::vector<float> actions(num_dofs, 0.0f);
    const char* env = std::getenv("RL_PARKOUR_JOINT_TEST");
    if (!env) return actions;

    const std::string mode(env);
    int index = 0;
    if (mode == "sweep")
    {
        index = (step_count / 100) % num_dofs;
    }
    else
    {
        try
        {
            index = std::stoi(mode);
        }
        catch (...)
        {
            index = 0;
        }
    }
    index = std::clamp(index, 0, num_dofs - 1);
    const float phase = static_cast<float>(step_count) * 0.12f;
    actions[index] = 0.8f * std::sin(phase);
    return actions;
}

static bool parkour_joint_test_enabled()
{
    return std::getenv("RL_PARKOUR_JOINT_TEST") != nullptr;
}

static void maybe_print_once(const char* message)
{
    static std::mutex mutex;
    static std::set<std::string> printed;
    std::lock_guard<std::mutex> lock(mutex);
    if (printed.insert(message).second)
    {
        std::cout << LOGGER::INFO << message << std::endl;
    }
}

}


RL_Sim::RL_Sim(int argc, char **argv)
{
#if defined(USE_ROS1)
    this->ang_vel_axis = "world";
    ros::NodeHandle nh;
    nh.param<std::string>("ros_namespace", this->ros_namespace, "");
    nh.param<std::string>("robot_name", this->robot_name, "");
#elif defined(USE_ROS2)
    ros2_node = std::make_shared<rclcpp::Node>("rl_sim_node");
    this->ang_vel_axis = "body";
    this->ros_namespace = ros2_node->get_namespace();
    // get params from param_node
    param_client = ros2_node->create_client<rcl_interfaces::srv::GetParameters>("/param_node/get_parameters");
    int retry_count = 0;
    while (!param_client->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok()) {
            std::cout << LOGGER::ERROR << "Interrupted while waiting for param_node service. Exiting." << std::endl;
            return;
        }
        if (++retry_count > 30) {
            std::cout << LOGGER::ERROR << "Timeout waiting for param_node service. Starting without remote params." << std::endl;
            break;
        }
        std::cout << LOGGER::WARNING << "Waiting for param_node service to be available..." << std::endl;
    }
    auto request = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
    request->names = {"robot_name", "gazebo_model_name"};
    auto future = param_client->async_send_request(request);
    auto status = rclcpp::spin_until_future_complete(ros2_node->get_node_base_interface(), future, std::chrono::seconds(5));
    if (status == rclcpp::FutureReturnCode::SUCCESS)
    {
        auto result = future.get();
        if (result->values.size() < 2)
        {
            std::cout << LOGGER::ERROR << "Failed to get all parameters from param_node" << std::endl;
        }
        else
        {
            this->robot_name = result->values[0].string_value;
            this->gazebo_model_name = result->values[1].string_value;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "Failed to call param_node service" << std::endl;
    }
    // fallback: use env var or default
    if (this->robot_name.empty())
    {
        const char* env = std::getenv("RL_SAR_ROBOT_NAME");
        this->robot_name = env ? env : "go2";
        std::cout << LOGGER::INFO << "Using robot_name (fallback): " << this->robot_name << std::endl;
    }
    if (this->gazebo_model_name.empty())
    {
        this->gazebo_model_name = this->robot_name + "_gazebo";
        std::cout << LOGGER::INFO << "Using gazebo_model_name (fallback): " << this->gazebo_model_name << std::endl;
    }
#endif

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");

    // depth debug save
    if (const char* env = std::getenv("RL_DEPTH_SAVE_DIR"))
    {
        this->depth_save_dir_ = env;
        std::filesystem::create_directories(this->depth_save_dir_);
        std::cout << LOGGER::INFO << "Depth debug save enabled: " << this->depth_save_dir_ << std::endl;
    }

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
#if defined(USE_ROS1)
    this->joint_publishers_commands.resize(this->params.Get<int>("num_of_dofs"));
#elif defined(USE_ROS2)
    this->robot_command_publisher_msg.motor_command.resize(this->params.Get<int>("num_of_dofs"));
    this->robot_state_subscriber_msg.motor_state.resize(this->params.Get<int>("num_of_dofs"));
#endif
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

#if defined(USE_ROS1)
    auto joint_controller_names_vec = this->params.Get<std::vector<std::string>>("joint_controller_names");  // avoid dangling reference
    this->StartJointController(this->ros_namespace, joint_controller_names_vec);
    // publisher
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/command";
        this->joint_publishers[joint_controller_name] =
            nh.advertise<robot_msgs::MotorCommand>(topic_name, 10);
    }

    // subscriber
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Sim::CmdvelCallback, this);
    this->joy_subscriber = nh.subscribe<sensor_msgs::Joy>("/joy", 10, &RL_Sim::JoyCallback, this);
    this->model_state_subscriber = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, &RL_Sim::ModelStatesCallback, this);
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/state";
        this->joint_subscribers[joint_controller_name] =
            nh.subscribe<robot_msgs::MotorState>(topic_name, 10,
                [this, joint_controller_name](const robot_msgs::MotorState::ConstPtr &msg)
                {
                    this->JointStatesCallback(msg, joint_controller_name);
                }
            );
        this->joint_positions[joint_controller_name] = 0.0f;
        this->joint_velocities[joint_controller_name] = 0.0f;
        this->joint_efforts[joint_controller_name] = 0.0f;
    }

    // service
    nh.param<std::string>("gazebo_model_name", this->gazebo_model_name, "");
    this->gazebo_pause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    this->gazebo_unpause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    this->gazebo_reset_world_client = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
#elif defined(USE_ROS2)
    this->StartJointController(this->ros_namespace, this->params.Get<std::vector<std::string>>("joint_names"));
    // publisher
    this->robot_command_publisher = ros2_node->create_publisher<robot_msgs::msg::RobotCommand>(
        this->ros_namespace + "robot_joint_controller/command", rclcpp::SystemDefaultsQoS());

    // subscriber
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
    this->joy_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", rclcpp::SystemDefaultsQoS(),
        [this] (const sensor_msgs::msg::Joy::SharedPtr msg) {this->JoyCallback(msg);}
    );
    this->gazebo_imu_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SystemDefaultsQoS(), [this] (const sensor_msgs::msg::Imu::SharedPtr msg) {this->GazeboImuCallback(msg);}
    );
    this->robot_state_subscriber = ros2_node->create_subscription<robot_msgs::msg::RobotState>(
        this->ros_namespace + "robot_joint_controller/state", rclcpp::SystemDefaultsQoS(),
        [this] (const robot_msgs::msg::RobotState::SharedPtr msg) {this->RobotStateCallback(msg);}
    );
    this->depth_sub_ = ros2_node->create_subscription<sensor_msgs::msg::Image>(
        "/depth_camera/depth/image_raw", rclcpp::SensorDataQoS(),
        [this] (const sensor_msgs::msg::Image::SharedPtr msg) {this->DepthImageCallback(msg);}
    );
    const std::vector<std::string> contact_topics = {
        "/foot_contacts/FL",
        "/foot_contacts/FR",
        "/foot_contacts/RL",
        "/foot_contacts/RR",
    };
    this->foot_contact_subscribers.reserve(contact_topics.size());
    for (size_t i = 0; i < contact_topics.size(); ++i)
    {
        this->foot_contact_subscribers.push_back(
            ros2_node->create_subscription<gazebo_msgs::msg::ContactsState>(
                contact_topics[i], rclcpp::SensorDataQoS(),
                [this, i] (const gazebo_msgs::msg::ContactsState::SharedPtr msg)
                {
                    this->FootContactCallback(msg, static_cast<int>(i));
                }
            )
        );
    }

    // service
    this->gazebo_pause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/pause_physics");
    this->gazebo_unpause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/unpause_physics");
    this->gazebo_reset_world_client = ros2_node->create_client<std_srvs::srv::Empty>("/reset_world");

    auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
    auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
}

RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names)
{
#if defined(USE_ROS1)
    pid_t pid0 = fork();
    if (pid0 == 0)
    {
        std::string cmd = "rosrun controller_manager spawner joint_state_controller ";
        for (const auto& name : names)
        {
            cmd += name + " ";
        }
        cmd += "__ns:=" + ros_namespace;
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
#elif defined(USE_ROS2)
    const char* ros_distro = std::getenv("ROS_DISTRO");
    std::string spawner = (ros_distro && std::string(ros_distro) == "foxy") ? "spawner.py" : "spawner";

    std::filesystem::path tmp_path = std::filesystem::temp_directory_path() / "robot_joint_controller_params.yaml";
    {
        std::ofstream tmp_file(tmp_path);
        if (!tmp_file)
        {
            throw std::runtime_error("Failed to create temporary parameter file");
        }

        tmp_file << "/robot_joint_controller:\n";
        tmp_file << "    ros__parameters:\n";
        tmp_file << "        joints:\n";
        for (const auto& name : names)
        {
            tmp_file << "            - " << name << "\n";
        }
    }

    pid_t pid = fork();
    if (pid == 0)
    {
        std::string cmd = "ros2 run controller_manager " + spawner + " robot_joint_controller ";
        cmd += "--controller-manager-timeout 15 ";
        cmd += "-p " + tmp_path.string() + " ";
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
    else if (pid > 0)
    {
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
        {
            std::cout << LOGGER::WARNING << "Joint controller spawner exited with status " << WEXITSTATUS(status)
                      << " (may already be loaded)" << std::endl;
        }

        std::filesystem::remove(tmp_path);
    }
    else
    {
        throw std::runtime_error("fork() failed");
    }
#endif
}

void RL_Sim::GetState(RobotState<float> *state)
{
#if defined(USE_ROS1)
    const auto &orientation = this->pose.orientation;
    const auto &angular_velocity = this->vel.angular;
#elif defined(USE_ROS2)
    const auto &orientation = this->gazebo_imu.orientation;
    const auto &angular_velocity = this->gazebo_imu.angular_velocity;
#endif

    state->imu.quaternion[0] = orientation.w;
    state->imu.quaternion[1] = orientation.x;
    state->imu.quaternion[2] = orientation.y;
    state->imu.quaternion[3] = orientation.z;

    state->imu.gyroscope[0] = angular_velocity.x;
    state->imu.gyroscope[1] = angular_velocity.y;
    state->imu.gyroscope[2] = angular_velocity.z;

#if defined(USE_ROS2)
    const auto &lin_accel = this->gazebo_imu.linear_acceleration;
    state->imu.accelerometer[0] = lin_accel.x;
    state->imu.accelerometer[1] = lin_accel.y;
    state->imu.accelerometer[2] = lin_accel.z;
#endif

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        state->motor_state.q[i] = this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.dq[i] = this->joint_velocities[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
#elif defined(USE_ROS2)
        state->motor_state.q[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].q;
        state->motor_state.dq[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq;
        state->motor_state.tau_est[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
#endif
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
#elif defined(USE_ROS2)
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
#endif
    }

#if defined(USE_ROS1)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->joint_publishers[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]].publish(this->joint_publishers_commands[i]);
    }
#elif defined(USE_ROS2)
    this->robot_command_publisher->publish(this->robot_command_publisher_msg);
#endif
}

void RL_Sim::RobotControl()
{
    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
#if defined(USE_ROS1)
        std_srvs::Empty empty;
        this->gazebo_reset_world_client.call(empty);
#elif defined(USE_ROS2)
        auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
        auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_pause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_pause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_unpause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_unpause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    this->control.ClearInput();

    // Termination check for parkour: reset if roll/pitch too high or base too low
    if (this->config_name == "parkour" && this->rl_init_done && simulation_running)
    {
        std::vector<float> euler = QuaternionToEuler(this->robot_state.imu.quaternion);
        float roll = euler[0];
        float pitch = euler[1];
        while (roll > M_PI) roll -= 2.0f * M_PI;
        while (roll < -M_PI) roll += 2.0f * M_PI;
        while (pitch > M_PI) pitch -= 2.0f * M_PI;
        while (pitch < -M_PI) pitch += 2.0f * M_PI;

        if (std::abs(roll) > 1.5f || std::abs(pitch) > 1.5f)
        {
            std::cout << std::endl << LOGGER::WARNING << "Parkour termination: roll=" << roll
                      << " pitch=" << pitch << " — resetting" << std::endl;
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_reset_world_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif
            this->ResetRuntimeState();
            this->control.x = 0.0f;
            this->control.y = 0.0f;
            this->control.yaw = 0.0f;
            this->rl_init_done = false;
            this->fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    this->SetCommand(&this->robot_command);
}

void RL_Sim::ResetRuntimeState()
{
    this->episode_length_buf = 0;
    this->parkour_step_count = 0;
    this->parkour_delta_yaw = 0.0f;
    this->parkour_prev_yaw = 0.0f;
    this->parkour_obs_history.clear();
    this->parkour_prev_action.clear();
    this->parkour_action_delay_buf.clear();
    this->parkour_scandots_latent.clear();
    this->parkour_depth_yaw_[0] = 0.0f;
    this->parkour_depth_yaw_[1] = 0.0f;

    {
        std::lock_guard<std::mutex> lock(this->foot_contact_mutex_);
        std::fill(this->foot_contacts_.begin(), this->foot_contacts_.end(), -0.5f);
        std::fill(this->foot_contacts_raw_.begin(), this->foot_contacts_raw_.end(), -0.5f);
        std::fill(this->foot_contacts_prev_raw_.begin(), this->foot_contacts_prev_raw_.end(), -0.5f);
    }

    {
        std::lock_guard<std::mutex> lock(this->depth_mutex_);
        this->depth_frame_buf_.clear();
        this->depth_frame_prev_processed_.clear();
        this->depth_frame_latest_.clear();
    }
    this->depth_encoder_.reset();
    this->use_depth_ = false;

    std::vector<float> discard;
    while (this->output_dof_pos_queue.try_pop(discard)) {}
    while (this->output_dof_vel_queue.try_pop(discard)) {}
    while (this->output_dof_tau_queue.try_pop(discard)) {}
}

#if defined(USE_ROS1)
void RL_Sim::ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg)
{
    this->vel = msg->twist[2];
    this->pose = msg->pose[2];
}
#elif defined(USE_ROS2)
void RL_Sim::GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    this->gazebo_imu = *msg;
}

void RL_Sim::FootContactCallback(const gazebo_msgs::msg::ContactsState::SharedPtr msg, int foot_idx)
{
    if (foot_idx < 0 || foot_idx >= 4) return;
    float max_force_norm = 0.0f;
    for (const auto& state : msg->states)
    {
        const auto& total_force = state.total_wrench.force;
        max_force_norm = std::max(max_force_norm, static_cast<float>(std::sqrt(
            total_force.x * total_force.x +
            total_force.y * total_force.y +
            total_force.z * total_force.z)));
        for (const auto& wrench : state.wrenches)
        {
            const auto& force = wrench.force;
            max_force_norm = std::max(max_force_norm, static_cast<float>(std::sqrt(
                force.x * force.x +
                force.y * force.y +
                force.z * force.z)));
        }
    }

    std::lock_guard<std::mutex> lock(this->foot_contact_mutex_);
    this->foot_contacts_raw_[foot_idx] = max_force_norm > 2.0f ? 0.5f : -0.5f;
}
#endif

void RL_Sim::UpdateParkourContactFilter()
{
    std::lock_guard<std::mutex> lock(this->foot_contact_mutex_);
    for (size_t i = 0; i < this->foot_contacts_.size(); ++i)
    {
        this->foot_contacts_[i] =
            (this->foot_contacts_raw_[i] > 0.0f || this->foot_contacts_prev_raw_[i] > 0.0f) ? 0.5f : -0.5f;
        this->foot_contacts_prev_raw_[i] = this->foot_contacts_raw_[i];
    }
}

void RL_Sim::CmdvelCallback(
#if defined(USE_ROS1)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}

void RL_Sim::JoyCallback(
#if defined(USE_ROS1)
    const sensor_msgs::Joy::ConstPtr &msg
#elif defined(USE_ROS2)
    const sensor_msgs::msg::Joy::SharedPtr msg
#endif
)
{
    this->joy_msg = *msg;

    // joystick control
    // Description of buttons and axes(F710):
    // |__ buttons[]: A=0, B=1, X=2, Y=3, LB=4, RB=5, back=6, start=7, power=8, stickL=9, stickR=10
    // |__ axes[]: Lx=0, Ly=1, Rx=3, Ry=4, LT=2, RT=5, DPadX=6, DPadY=7

    if (this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::A);
    if (this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::B);
    if (this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::X);
    if (this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->joy_msg.buttons[4]) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->joy_msg.axes[1]; // LY
    this->control.y = this->joy_msg.axes[0]; // LX
    this->control.yaw = this->joy_msg.axes[3]; // RX
}

#if defined(USE_ROS1)
void RL_Sim::JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name)
{
    this->joint_positions[joint_controller_name] = msg->q;
    this->joint_velocities[joint_controller_name] = msg->dq;
    this->joint_efforts[joint_controller_name] = msg->tau_est;
}
#elif defined(USE_ROS2)
void RL_Sim::RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
    this->robot_state_subscriber_msg = *msg;
}
#endif

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        if (env_flag_enabled("RL_GYRO_ZERO"))
        {
            maybe_print_once("Parkour gyro observation zeroed by RL_GYRO_ZERO");
            this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
        }
        else
        {
            if (env_flag_enabled("RL_GYRO_FLIP_X"))
            {
                maybe_print_once("Parkour gyro X flipped by RL_GYRO_FLIP_X");
                this->obs.ang_vel[0] *= -1.0f;
            }
            if (env_flag_enabled("RL_GYRO_FLIP_Y"))
            {
                maybe_print_once("Parkour gyro Y flipped by RL_GYRO_FLIP_Y");
                this->obs.ang_vel[1] *= -1.0f;
            }
            if (env_flag_enabled("RL_GYRO_FLIP_Z"))
            {
                maybe_print_once("Parkour gyro Z flipped by RL_GYRO_FLIP_Z");
                this->obs.ang_vel[2] *= -1.0f;
            }
        }
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        if (this->config_name == "parkour")
        {
            this->obs.commands[0] = parkour_command_x(this->control.x);
        }
        if (this->control.navigation_mode)
        {
            this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        // IsaacLab refreshes delta_yaw and delta_next_yaw every 5 policy steps.
        if (this->config_name == "parkour")
        {
            this->parkour_step_count++;
            this->UpdateParkourContactFilter();
            if (this->parkour_step_count % DEPTH_ENC_INTERVAL == 0)
            {
                std::vector<float> euler = QuaternionToEuler(this->obs.base_quat);
                float yaw = euler[2];
                while (yaw > M_PI) yaw -= 2.0f * M_PI;
                while (yaw < -M_PI) yaw += 2.0f * M_PI;
                this->parkour_delta_yaw = yaw;
                this->parkour_prev_yaw = yaw;
            }
        }

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);
        if (this->config_name == "parkour" && this->parkour_step_count % 50 == 0)
        {
            auto print_leg = [&](const char* name, int hip, int thigh, int calf)
            {
                std::cout << " " << name
                          << " q:[" << this->obs.dof_pos[hip] << "," << this->obs.dof_pos[thigh] << "," << this->obs.dof_pos[calf] << "]"
                          << " tgt:[" << this->output_dof_pos[hip] << "," << this->output_dof_pos[thigh] << "," << this->output_dof_pos[calf] << "]"
                          << " act:[" << this->obs.actions[hip] << "," << this->obs.actions[thigh] << "," << this->obs.actions[calf] << "]";
            };
            std::cout << LOGGER::INFO << "Parkour joint debug";
            print_leg("FL", 0, 4, 8);
            print_leg("FR", 1, 5, 9);
            print_leg("RL", 2, 6, 10);
            print_leg("RR", 3, 7, 11);
            std::cout << std::endl;
        }

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    // Parkour config uses custom observation building (proprio + scan + priv + latent + history)
    if (this->config_name == "parkour")
    {
        if (parkour_joint_test_enabled())
        {
            std::vector<float> actions = build_parkour_joint_test_action(
                this->parkour_step_count, this->params.Get<int>("num_of_dofs"));
            if (this->parkour_step_count % 50 == 0)
            {
                static const std::vector<std::string> policy_joint_names = {
                    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
                    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
                    "FL_calf", "FR_calf", "RL_calf", "RR_calf"
                };
                auto action_minmax = std::minmax_element(actions.begin(), actions.end());
                int active_index = static_cast<int>(std::distance(
                    actions.begin(), std::max_element(actions.begin(), actions.end(),
                        [](float a, float b) { return std::abs(a) < std::abs(b); })));
                const auto controller_names = this->params.Get<std::vector<std::string>>("joint_controller_names");
                const auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
                const int gazebo_index = joint_mapping[active_index];
                std::cout << LOGGER::INFO
                          << "JointTest policy_idx:" << active_index
                          << " policy_joint:" << policy_joint_names[active_index]
                          << " gazebo_idx:" << gazebo_index
                          << " controller:" << controller_names[gazebo_index]
                          << " action:" << actions[active_index]
                          << " action_min:" << *action_minmax.first
                          << " action_max:" << *action_minmax.second
                          << std::endl;
            }
            return actions;
        }

        const bool disable_depth = env_flag_enabled("RL_PARKOUR_DISABLE_DEPTH");
        if (disable_depth)
        {
            maybe_print_once("Parkour depth encoder disabled by RL_PARKOUR_DISABLE_DEPTH");
            this->use_depth_ = false;
            this->parkour_depth_yaw_[0] = 0.0f;
            this->parkour_depth_yaw_[1] = 0.0f;
            const int latent_dim = this->params.Get<int>("scandots_latent_dim");
            this->parkour_scandots_latent.assign(latent_dim, 0.0f);
        }

        // Lazy-init depth encoder on first call
        if (!disable_depth && !this->use_depth_ && !this->depth_encoder_)
        {
            this->use_depth_ = this->params.Get<bool>("use_depth");
            if (this->use_depth_) this->InitDepthEncoder();
        }

        // Update depth encoding every N steps
        const bool freeze_depth_latent =
            env_flag_enabled("RL_PARKOUR_FREEZE_DEPTH_LATENT") && !this->parkour_scandots_latent.empty();
        if (freeze_depth_latent)
        {
            maybe_print_once("Parkour depth latent frozen by RL_PARKOUR_FREEZE_DEPTH_LATENT");
        }
        if (!disable_depth && !freeze_depth_latent && this->use_depth_ && this->parkour_step_count % DEPTH_ENC_INTERVAL == 0 && this->parkour_step_count > 0)
        {
            std::vector<float> proprio = this->BuildParkourProprio();
            // Zero yaw slots [6:8] before depth encoder (match training)
            proprio[6] = 0.0f;
            proprio[7] = 0.0f;
            this->UpdateDepthEncoding(proprio);
        }

        std::vector<float> full_obs = this->BuildParkourObs();

        if (this->parkour_scandots_latent.empty())
        {
            this->parkour_scandots_latent.resize(this->params.Get<int>("scandots_latent_dim"), 0.0f);
        }

        // Flat Gazebo has no parkour target yaw. Keep yaw slots zero by default.
        // Set RL_PARKOUR_USE_DEPTH_YAW=1 to match IsaacLab demo.py:
        //   obs[:, 6:8] = 1.5 * yaw
        full_obs[6] = 0.0f;
        full_obs[7] = 0.0f;
        if (!disable_depth && this->use_depth_ && env_flag_enabled("RL_PARKOUR_USE_DEPTH_YAW"))
        {
            maybe_print_once("Parkour depth yaw enabled by RL_PARKOUR_USE_DEPTH_YAW");
            full_obs[6] = 1.5f * this->parkour_depth_yaw_[0];
            full_obs[7] = 1.5f * this->parkour_depth_yaw_[1];
        }

        std::vector<float> actions = this->model->forward({full_obs, this->parkour_scandots_latent});
        if (!all_finite(actions))
        {
            static bool warned_nonfinite_action = false;
            if (!warned_nonfinite_action)
            {
                std::cout << LOGGER::WARNING << "Parkour policy returned non-finite actions; using zero action" << std::endl;
                warned_nonfinite_action = true;
            }
            actions.assign(this->params.Get<int>("num_of_dofs"), 0.0f);
        }
        if (this->parkour_step_count % 50 == 0)
        {
            auto action_minmax = std::minmax_element(actions.begin(), actions.end());
            std::vector<float> contacts(4, -0.5f);
            {
                std::lock_guard<std::mutex> lock(this->foot_contact_mutex_);
                contacts = this->foot_contacts_;
            }
            std::vector<float> euler = QuaternionToEuler(this->obs.base_quat);
            std::cout << LOGGER::INFO
                      << "Parkour debug"
                      << " cmd_x:" << parkour_command_x(this->control.x)
                      << " roll:" << euler[0]
                      << " pitch:" << euler[1]
                      << " gyro:[" << this->obs.ang_vel[0] << "," << this->obs.ang_vel[1] << "," << this->obs.ang_vel[2] << "]"
                      << " depth_yaw:[" << this->parkour_depth_yaw_[0] << "," << this->parkour_depth_yaw_[1] << "]"
                      << " contacts:[" << contacts[0] << "," << contacts[1] << "," << contacts[2] << "," << contacts[3] << "]"
                      << " hip_lr:[" << (actions[0] + actions[6]) * 0.5f << "," << (actions[3] + actions[9]) * 0.5f << "]"
                      << " thigh_lr:[" << (actions[1] + actions[7]) * 0.5f << "," << (actions[4] + actions[10]) * 0.5f << "]"
                      << " calf_lr:[" << (actions[2] + actions[8]) * 0.5f << "," << (actions[5] + actions[11]) * 0.5f << "]"
                      << " action_min:" << (actions.empty() ? 0.0f : *action_minmax.first)
                      << " action_max:" << (actions.empty() ? 0.0f : *action_minmax.second)
                      << std::endl;
        }

        // IsaacLab DelayedJointPositionAction: actions are delayed by 1 step.
        // First step: delayed action = zeros (robot holds default_dof_pos).
        int num_dofs = this->params.Get<int>("num_of_dofs");
        if (this->parkour_action_delay_buf.empty())
            this->parkour_action_delay_buf.resize(num_dofs * 2, 0.0f);
        // Shift: prev <- curr, then store new actions as curr
        std::copy(this->parkour_action_delay_buf.begin() + num_dofs, this->parkour_action_delay_buf.end(),
                  this->parkour_action_delay_buf.begin());
        std::copy(actions.begin(), actions.end(), this->parkour_action_delay_buf.begin() + num_dofs);
        // Use delayed action (prev slot)
        std::vector<float> delayed_actions(this->parkour_action_delay_buf.begin(),
                                           this->parkour_action_delay_buf.begin() + num_dofs);
        if (env_flag_enabled("RL_PARKOUR_NO_ACTION_DELAY"))
        {
            maybe_print_once("Parkour action delay disabled by RL_PARKOUR_NO_ACTION_DELAY");
            delayed_actions = actions;
        }

        // IsaacLab: observation gets raw (undelayed) action, but robot gets delayed action.
        // parkour_prev_action stores the CURRENT raw output for the next obs build.
        this->parkour_prev_action = actions;   // raw policy output for next observation
        // delayed_actions is what gets applied to the robot (1-step delay)

        // Update observation history (with yaw indices zeroed, matching IsaacLab)
        std::vector<float> proprio_for_hist = this->BuildParkourProprio();
        proprio_for_hist[6] = 0.0f;
        proprio_for_hist[7] = 0.0f;
        this->UpdateParkourHistory(proprio_for_hist);

        if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
        {
            delayed_actions = clamp(delayed_actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
        }
        return delayed_actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
#if defined(USE_ROS1)
        this->plot_real_joint_pos[i].push_back(this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]]);
        this->plot_target_joint_pos[i].push_back(this->joint_publishers_commands[i].q);
#elif defined(USE_ROS2)
        this->plot_real_joint_pos[i].push_back(this->robot_state_subscriber_msg.motor_state[i].q);
        this->plot_target_joint_pos[i].push_back(this->robot_command_publisher_msg.motor_command[i].q);
#endif
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

std::vector<float> RL_Sim::BuildParkourProprio()
{
    std::vector<float> default_dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    std::vector<float> proprio;
    proprio.reserve(53);

    // 1. Body angular velocity * 0.25 (3 dims)
    for (int i = 0; i < 3; i++)
        proprio.push_back(finite_or_zero(this->obs.ang_vel[i]) * 0.25f);

    // 2. Roll, Pitch from quaternion (2 dims)
    std::vector<float> euler = QuaternionToEuler(this->obs.base_quat);
    float roll = euler[0];
    float pitch = euler[1];
    while (roll > M_PI) roll -= 2.0f * M_PI;
    while (roll < -M_PI) roll += 2.0f * M_PI;
    while (pitch > M_PI) pitch -= 2.0f * M_PI;
    while (pitch < -M_PI) pitch += 2.0f * M_PI;
    proprio.push_back(finite_or_zero(roll));
    proprio.push_back(finite_or_zero(pitch));

    // 3. 0 * delta_yaw (1 dim)
    proprio.push_back(0.0f);

    // 4. delta_yaw = wrap_to_pi(current_yaw) (1 dim)
    proprio.push_back(finite_or_zero(this->parkour_delta_yaw));

    // 5. delta_next_yaw = same as delta_yaw (1 dim)
    proprio.push_back(finite_or_zero(this->parkour_delta_yaw));

    // 6. 0 * commands[0:2] (2 dims)
    proprio.push_back(0.0f);
    proprio.push_back(0.0f);

    // 7. commands[0]: lin_vel_x = control.x (1 dim)
    proprio.push_back(parkour_command_x(this->control.x));

    // 8. env_idx = 1.0 for non-flat terrain, 0.0 for flat (1 dim)
    proprio.push_back(0.0f);

    // 9. invert_env_idx = 1.0 for flat terrain, 0.0 for non-flat (1 dim)
    proprio.push_back(1.0f);

    // 10. joint_pos - default_dof_pos (12 dims)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); i++)
        proprio.push_back(finite_or_zero(this->obs.dof_pos[i]) - default_dof_pos[i]);

    // 11. joint_vel * 0.05 (12 dims)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); i++)
        proprio.push_back(finite_or_zero(this->obs.dof_vel[i]) * 0.05f);

    // 12. previous action (12 dims)
    if (this->parkour_prev_action.empty())
        this->parkour_prev_action.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); i++)
        proprio.push_back(finite_or_zero(this->parkour_prev_action[i]));

    // 13. Contact fill (4 dims), IsaacLab order: FL, FR, RL, RR.
    std::vector<float> contacts(4, -0.5f);
    {
        std::lock_guard<std::mutex> lock(this->foot_contact_mutex_);
        contacts = this->foot_contacts_;
    }
    for (int i = 0; i < 4; i++)
        proprio.push_back(contacts[i]);

    return proprio;
}

void RL_Sim::UpdateParkourHistory(const std::vector<float>& proprio)
{
    int history_length = this->params.Get<int>("history_length");
    int num_prop = this->params.Get<int>("num_prop");
    int hist_size = history_length * num_prop;

    if (this->parkour_obs_history.empty())
    {
        this->parkour_obs_history.resize(hist_size, 0.0f);
        for (int h = 0; h < history_length; h++)
        {
            for (int i = 0; i < num_prop; i++)
            {
                this->parkour_obs_history[h * num_prop + i] = proprio[i];
            }
        }
    }
    else
    {
        for (int h = 0; h < history_length - 1; h++)
        {
            for (int i = 0; i < num_prop; i++)
            {
                this->parkour_obs_history[h * num_prop + i] =
                    this->parkour_obs_history[(h + 1) * num_prop + i];
            }
        }
        for (int i = 0; i < num_prop; i++)
        {
            this->parkour_obs_history[(history_length - 1) * num_prop + i] = proprio[i];
        }
    }
}

std::vector<float> RL_Sim::BuildParkourObs()
{
    int num_prop = this->params.Get<int>("num_prop");
    int num_scan = this->params.Get<int>("num_scan");
    int num_priv_explicit = this->params.Get<int>("num_priv_explicit");
    int num_priv_latent = this->params.Get<int>("num_priv_latent");
    int history_length = this->params.Get<int>("history_length");

    std::vector<float> proprio = this->BuildParkourProprio();

    int total_size = num_prop + num_scan + num_priv_explicit + num_priv_latent + history_length * num_prop;
    std::vector<float> obs;
    obs.reserve(total_size);

    // Proprio (53 dims)
    obs.insert(obs.end(), proprio.begin(), proprio.end());

    // Height scan (132 dims) - dummy zeros
    for (int i = 0; i < num_scan; i++)
        obs.push_back(0.0f);

    // Priv explicit (9 dims) - estimated by model's internal estimator
    for (int i = 0; i < num_priv_explicit; i++)
        obs.push_back(0.0f);

    // Priv latent (29 dims) - zeros
    for (int i = 0; i < num_priv_latent; i++)
        obs.push_back(0.0f);

    // History buffer (history_length * num_prop = 530 dims)
    if (this->parkour_obs_history.empty())
        this->parkour_obs_history.resize(history_length * num_prop, 0.0f);
    obs.insert(obs.end(), this->parkour_obs_history.begin(), this->parkour_obs_history.end());

    return obs;
}

void RL_Sim::InitDepthEncoder()
{
    std::string depth_path = std::string(POLICY_DIR) + "/" + this->robot_name + "/parkour/depth_latest.pt";
    if (!std::filesystem::exists(depth_path))
    {
        std::cout << LOGGER::WARNING << "Depth encoder not found at " << depth_path << ", disabling depth" << std::endl;
        this->use_depth_ = false;
        return;
    }

    try
    {
        this->depth_encoder_ = InferenceRuntime::ModelFactory::load_model(depth_path);
        if (this->depth_encoder_ && this->depth_encoder_->is_loaded())
        {
            this->use_depth_ = true;
            std::cout << LOGGER::INFO << "Depth encoder loaded: " << depth_path << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::WARNING << "Failed to load depth encoder: " << e.what() << std::endl;
        this->use_depth_ = false;
    }
}

void RL_Sim::DepthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // Only process depth for parkour.
    if (this->config_name != "parkour" || !this->use_depth_) return;

    if (msg->width != DEPTH_RAW_W || msg->height != DEPTH_RAW_H) return;

    const int raw_size = DEPTH_RAW_W * DEPTH_RAW_H;
    std::vector<float> raw_depth(raw_size, DEPTH_CLIP_MAX);
    if (msg->encoding == "32FC1")
    {
        if (msg->data.size() < static_cast<size_t>(raw_size * sizeof(float))) return;
        std::memcpy(raw_depth.data(), msg->data.data(), raw_size * sizeof(float));
    }
    else if (msg->encoding == "16UC1")
    {
        if (msg->data.size() < static_cast<size_t>(raw_size * sizeof(uint16_t))) return;
        const auto* raw_u16 = reinterpret_cast<const uint16_t*>(msg->data.data());
        for (int i = 0; i < raw_size; ++i)
        {
            raw_depth[i] = static_cast<float>(raw_u16[i]) * 0.001f;
        }
    }
    else
    {
        static bool warned = false;
        if (!warned)
        {
            std::cout << LOGGER::WARNING << "Unsupported depth image encoding: " << msg->encoding << std::endl;
            warned = true;
        }
        return;
    }

    std::lock_guard<std::mutex> lock(this->depth_mutex_);
    this->depth_frame_latest_ = std::move(raw_depth);
}

void RL_Sim::ProcessAndStoreDepth(const float* raw_depth)
{
    // 1. Gazebo depth is image-plane z-depth. Convert it to ray distance first.
    const int raw_size = DEPTH_RAW_W * DEPTH_RAW_H;
    std::vector<float> ray_depth(raw_size);
    for (int v = 0; v < DEPTH_RAW_H; ++v)
    {
        for (int u = 0; u < DEPTH_RAW_W; ++u)
        {
            const int i = v * DEPTH_RAW_W + u;
            ray_depth[i] = image_plane_depth_to_ray_depth(sanitize_depth(raw_depth[i]), u, v);
        }
    }

    // 2. Clip to IsaacLab RayCaster max_distance.
    std::vector<float> clipped(raw_size);
    for (int i = 0; i < raw_size; ++i)
    {
        clipped[i] = std::clamp(ray_depth[i], 0.0f, DEPTH_CLIP_MAX);
    }

    // 3. Crop: height[:-2], width[4:-4]. Raw is H=60, W=106; crop is H=58, W=98.
    const int crop_h = DEPTH_RAW_H - 2;
    const int crop_w = DEPTH_RAW_W - 8;
    std::vector<float> cropped(crop_h * crop_w);
    for (int h = 0; h < crop_h; ++h)
    {
        for (int w = 0; w < crop_w; ++w)
        {
            cropped[h * crop_w + w] = clipped[h * DEPTH_RAW_W + (w + 4)];
        }
    }

    // 4. Resize with bicubic interpolation to H=58, W=87.
    const int out_size = DEPTH_H * DEPTH_W;
    std::vector<float> resized(out_size);
    bicubic_resize(cropped.data(), crop_h, crop_w, resized.data(), DEPTH_H, DEPTH_W);

    if (env_flag_enabled("RL_DEPTH_FLIP_HORIZONTAL"))
    {
        maybe_print_once("Depth preprocessing: horizontal flip enabled by RL_DEPTH_FLIP_HORIZONTAL");
        for (int h = 0; h < DEPTH_H; ++h)
        {
            for (int w = 0; w < DEPTH_W / 2; ++w)
            {
                std::swap(resized[h * DEPTH_W + w], resized[h * DEPTH_W + (DEPTH_W - 1 - w)]);
            }
        }
    }

    if (env_flag_enabled("RL_DEPTH_FLIP_VERTICAL"))
    {
        maybe_print_once("Depth preprocessing: vertical flip enabled by RL_DEPTH_FLIP_VERTICAL");
        for (int h = 0; h < DEPTH_H / 2; ++h)
        {
            for (int w = 0; w < DEPTH_W; ++w)
            {
                std::swap(resized[h * DEPTH_W + w], resized[(DEPTH_H - 1 - h) * DEPTH_W + w]);
            }
        }
    }

    // 5. Normalize to match IsaacLab: depth / max_distance - 0.5.
    std::vector<float> processed(out_size);
    for (int i = 0; i < out_size; ++i)
    {
        const float depth = std::clamp(sanitize_depth(resized[i]), 0.0f, DEPTH_CLIP_MAX);
        processed[i] = depth / DEPTH_CLIP_MAX - 0.5f;
    }

    // 6. Debug save if enabled
    if (!this->depth_save_dir_.empty() && this->depth_save_counter_++ % DEPTH_SAVE_INTERVAL == 0)
        SaveDepthDebug(raw_depth, ray_depth.data(), cropped.data(), crop_h, crop_w, resized.data(), processed.data());

    if (this->depth_frame_prev_processed_.empty())
    {
        this->depth_frame_prev_processed_ = processed;
        this->depth_frame_buf_ = processed;
    }
    else
    {
        this->depth_frame_buf_ = this->depth_frame_prev_processed_;
        this->depth_frame_prev_processed_ = std::move(processed);
    }
}

static void save_pgm(const std::string& path, const float* data, int w, int h, float vmin, float vmax)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    f << "P5\n" << w << " " << h << "\n65535\n";
    for (int i = 0; i < w * h; ++i)
    {
        float val = (std::clamp(data[i], vmin, vmax) - vmin) / (vmax - vmin);
        uint16_t px = static_cast<uint16_t>(val * 65535.0f);
        // big-endian for PGM
        f.put(static_cast<char>(px >> 8));
        f.put(static_cast<char>(px & 0xFF));
    }
}

static void save_bin(const std::string& path, const float* data, int count)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
}

void RL_Sim::SaveDepthDebug(const float* raw, const float* ray,
                             const float* cropped, int crop_h, int crop_w,
                             const float* resized, const float* proc)
{
    char subdir[64];
    snprintf(subdir, sizeof(subdir), "step_%06d", this->depth_save_counter_);
    std::string dir = this->depth_save_dir_ + "/" + subdir;
    std::filesystem::create_directories(dir);

    // 1. Raw z-depth (60x106)
    save_pgm(dir + "/raw_depth.pgm", raw, DEPTH_RAW_W, DEPTH_RAW_H, 0.f, DEPTH_CLIP_MAX);
    save_bin(dir + "/raw_depth.bin",  raw, DEPTH_RAW_W * DEPTH_RAW_H);

    // 2. Ray depth after conversion (60x106)
    save_pgm(dir + "/ray_depth.pgm", ray, DEPTH_RAW_W, DEPTH_RAW_H, 0.f, DEPTH_CLIP_MAX);
    save_bin(dir + "/ray_depth.bin",  ray, DEPTH_RAW_W * DEPTH_RAW_H);

    // 3. Cropped (58x98), in meters
    save_pgm(dir + "/cropped_depth.pgm", cropped, crop_w, crop_h, 0.f, DEPTH_CLIP_MAX);
    save_bin(dir + "/cropped_depth.bin",  cropped, crop_h * crop_w);

    // 4. Resized (58x87), in meters
    save_pgm(dir + "/resized_depth.pgm", resized, DEPTH_W, DEPTH_H, 0.f, DEPTH_CLIP_MAX);
    save_bin(dir + "/resized_depth.bin",  resized, DEPTH_W * DEPTH_H);

    // 5. Processed (58x87), normalized [-0.5, 0.5]
    save_pgm(dir + "/processed_depth.pgm", proc, DEPTH_W, DEPTH_H, -0.5f, 0.5f);
    save_bin(dir + "/processed_depth.bin",  proc, DEPTH_W * DEPTH_H);

    std::cout << LOGGER::INFO << "[DepthSave] Saved debug frames to " << dir << std::endl;
}

void RL_Sim::UpdateDepthEncoding(const std::vector<float>& proprio_53)
{
    if (!this->depth_encoder_ || !this->depth_encoder_->is_loaded()) return;

    std::vector<float> raw_copy;
    {
        std::lock_guard<std::mutex> lock(this->depth_mutex_);
        if (this->depth_frame_latest_.empty()) return;
        raw_copy = this->depth_frame_latest_;
    }

    this->ProcessAndStoreDepth(raw_copy.data());

    // Run depth encoder: (depth[1,58,87], proprio[1,53]) -> [32+2]
    if (!this->depth_frame_buf_.empty())
    {
        try
        {
            std::vector<float> result = this->depth_encoder_->forward_with_shapes(
                {this->depth_frame_buf_, proprio_53},
                {{1, DEPTH_H, DEPTH_W}, {1, static_cast<int64_t>(proprio_53.size())}});
            if (result.size() >= 34 && all_finite(result))
            {
                // First 32: scandots_latent
                this->parkour_scandots_latent.assign(result.begin(), result.begin() + 32);
                // Last 2: yaw (delta_yaw, delta_next_yaw)
                this->parkour_depth_yaw_[0] = result[32];
                this->parkour_depth_yaw_[1] = result[33];
            }
            else if (result.size() >= 34)
            {
                static bool warned_nonfinite_depth = false;
                if (!warned_nonfinite_depth)
                {
                    std::cout << LOGGER::WARNING << "Depth encoder returned non-finite output; keeping previous latent" << std::endl;
                    warned_nonfinite_depth = true;
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::WARNING << "Depth encoder inference failed: " << e.what() << std::endl;
        }
    }
}

#if defined(USE_ROS1)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_ROS1)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Sim rl_sar(argc, argv);
    ros::spin();
#elif defined(USE_ROS2)
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Sim>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
#endif
    return 0;
}
