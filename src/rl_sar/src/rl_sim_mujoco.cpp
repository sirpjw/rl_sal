/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"

RL_Sim* RL_Sim::instance = nullptr;

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " robot_name scene_name" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");

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
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

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

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        state->imu.quaternion[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 0];
        state->imu.quaternion[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 1];
        state->imu.quaternion[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 2];
        state->imu.quaternion[3] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 3];

        state->imu.gyroscope[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 4];
        state->imu.gyroscope[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 5];
        state->imu.gyroscope[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 6];

        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            state->motor_state.q[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]];
            state->motor_state.dq[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")];
            state->motor_state.tau_est[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + 2 * this->params.Get<int>("num_of_dofs")];
        }
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            mj_data->ctrl[this->params.Get<std::vector<int>>("joint_mapping")[i]] =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]]) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")]);
        }
    }
}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);

    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    // Termination check for parkour: reset if roll/pitch too high or base too low
    if (this->config_name == "parkour" && this->rl_init_done && simulation_running)
    {
        std::vector<float> euler = QuaternionToEuler(this->robot_state.imu.quaternion);
        float roll = euler[0];
        float pitch = euler[1];
        // Wrap to [-pi, pi]
        while (roll > M_PI) roll -= 2.0f * M_PI;
        while (roll < -M_PI) roll += 2.0f * M_PI;
        while (pitch > M_PI) pitch -= 2.0f * M_PI;
        while (pitch < -M_PI) pitch += 2.0f * M_PI;

        float base_z = this->mj_data->qpos[2];  // base height
        if (std::abs(roll) > 1.5f || std::abs(pitch) > 1.5f || base_z < -0.25f)
        {
            std::cout << std::endl << LOGGER::WARNING << "Parkour termination: roll=" << roll
                      << " pitch=" << pitch << " height=" << base_z << " — resetting" << std::endl;
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
            this->episode_length_buf = 0;
            this->parkour_step_count = 0;
            this->parkour_delta_yaw = 0.0f;
            this->parkour_prev_yaw = 0.0f;
            this->parkour_obs_history.clear();
            this->parkour_prev_action.clear();
            // Reset joystick command on fall
            this->control.x = 0.0f;
            this->control.y = 0.0f;
            this->control.yaw = 0.0f;
        }
    }

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        // }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        // Track yaw for parkour
        if (this->config_name == "parkour")
        {
            // Compute yaw from quaternion
            std::vector<float> euler = QuaternionToEuler(this->obs.base_quat);
            float yaw = euler[2];
            // Wrap yaw to [-pi, pi] - matches Python: wrap_to_pi(yaw)
            while (yaw > M_PI) yaw -= 2.0f * M_PI;
            while (yaw < -M_PI) yaw += 2.0f * M_PI;
            this->parkour_delta_yaw = yaw;
            this->parkour_prev_yaw = yaw;
            this->parkour_step_count++;
        }

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

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

    // Check if using parkour config
    if (this->config_name == "parkour")
    {
        // Build full parkour observation (proprio + scan + priv + latent + history)
        std::vector<float> full_obs = this->BuildParkourObs();

        // scandots_latent: 32-dim zeros for sim (no depth camera)
        if (this->parkour_scandots_latent.empty())
        {
            this->parkour_scandots_latent.resize(this->params.Get<int>("scandots_latent_dim"), 0.0f);
        }

        // Call model with two inputs: (x, scandots_latent)
        std::vector<float> actions = this->model->forward({full_obs, this->parkour_scandots_latent});

        // Update history with proprio after model call
        std::vector<float> proprio = this->BuildParkourProprio();
        this->UpdateParkourHistory(proprio);

        // Save action for next step
        this->parkour_prev_action = actions;

        // Apply action clipping
        if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
        {
            return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
        }
        return actions;
    }
    else
    {
        // Original forward path for non-parkour configs
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
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
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
        proprio.push_back(this->obs.ang_vel[i] * 0.25f);

    // 2. Roll, Pitch from quaternion (2 dims)
    std::vector<float> euler = QuaternionToEuler(this->obs.base_quat);
    float roll = euler[0];
    float pitch = euler[1];
    // Wrap to [-pi, pi]
    while (roll > M_PI) roll -= 2.0f * M_PI;
    while (roll < -M_PI) roll += 2.0f * M_PI;
    while (pitch > M_PI) pitch -= 2.0f * M_PI;
    while (pitch < -M_PI) pitch += 2.0f * M_PI;
    proprio.push_back(roll);
    proprio.push_back(pitch);

    // 3. 0 * delta_yaw (1 dim)
    proprio.push_back(0.0f);

    // 4. delta_yaw (1 dim)
    proprio.push_back(this->parkour_delta_yaw);

    // 5. delta_next_yaw (1 dim) - same as delta_yaw for now
    proprio.push_back(this->parkour_delta_yaw);

    // 6. 0 * commands[0:2] (2 dims)
    proprio.push_back(0.0f);
    proprio.push_back(0.0f);

    // 7. commands[0]: lin_vel_x = control.x (1 dim)
    proprio.push_back(this->control.x);

    // 8. env_idx (1 dim) - always 1.0
    proprio.push_back(1.0f);

    // 9. invert_env_idx (1 dim) - always 0.0
    proprio.push_back(0.0f);

    // 10. joint_pos - default_dof_pos (12 dims)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); i++)
        proprio.push_back(this->obs.dof_pos[i] - default_dof_pos[i]);

    // 11. joint_vel * 0.05 (12 dims)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); i++)
        proprio.push_back(this->obs.dof_vel[i] * 0.05f);

    // 12. previous action (12 dims)
    if (this->parkour_prev_action.empty())
        this->parkour_prev_action.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); i++)
        proprio.push_back(this->parkour_prev_action[i]);

    // 13. Contact fill (4 dims) - dummy zeros for now
    for (int i = 0; i < 4; i++)
        proprio.push_back(-0.5f);  // no contact = -0.5

    return proprio;
}

void RL_Sim::UpdateParkourHistory(const std::vector<float>& proprio)
{
    int history_length = this->params.Get<int>("history_length");
    int num_prop = this->params.Get<int>("num_prop");
    int hist_size = history_length * num_prop;

    if (this->parkour_obs_history.empty())
    {
        // Initialize history buffer with repeated first observation
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
        // Shift history: drop oldest, append newest
        for (int h = 0; h < history_length - 1; h++)
        {
            for (int i = 0; i < num_prop; i++)
            {
                this->parkour_obs_history[h * num_prop + i] =
                    this->parkour_obs_history[(h + 1) * num_prop + i];
            }
        }
        // Append new proprio at the end
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

    // Compute proprio
    std::vector<float> proprio = this->BuildParkourProprio();

    // Build full observation: [proprio, height_scan, priv_explicit, priv_latent, history]
    int total_size = num_prop + num_scan + num_priv_explicit + num_priv_latent + history_length * num_prop;
    std::vector<float> obs;
    obs.reserve(total_size);

    // Proprio (53 dims)
    obs.insert(obs.end(), proprio.begin(), proprio.end());

    // Height scan (132 dims) - dummy zeros for now (no raycaster)
    for (int i = 0; i < num_scan; i++)
        obs.push_back(0.0f);

    // Priv explicit (9 dims) - will be estimated by model's internal estimator
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

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}
