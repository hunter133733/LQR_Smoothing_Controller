## Initial setup
Follow these instructions to set up the TurtleBot3.
1. Read the [TurtleBot3 Burger Manual](http://www.robotis.com/service/download.php?no=748&_gl=1*wc3mgb*_gcl_au*MjA1NzQ5Nzg5OS4xNzc0NjY5NDE3) to understand
    * The two ways of connecting power to the robot
    * How to use and care for the battery
    * How the different layers of the TurtleBot are connected
    * Safety precautions
2. Do not install any packages onto the TurtleBot3 unless approved by the instructor!
3. Attach a monitor via HDMI and power on the TurtleBot3, and connect the TurtleBot3 to wifi to enable SSH (this is way more convenient).
If you're not sure how to do this, see [this article](https://linuxconfig.org/ubuntu-20-04-connect-to-wifi-from-command-line).
(The article will also work for Ubuntu 22, which is installed on the TurtleBot3.)
4. Set the `LDS_MODEL` environment variable to specify the Lidar model according to the [official docs](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#sbc-setup)
5. Set Turtlebot3 model
```bash
export TURTLEBOT3_MODEL=burger
```
6. Try launching the main launch file (and also see the [official instrctions](https://emanual.robotis.com/docs/en/platform/turtlebot3/bringup/#bringup))
```bash
ros2 launch turtlebot3_bringup robot.launch.py
```
If the bringup launch file gives an error like `[turtlebot3_node]: Failed connection with Devices`, double check the physical connections with the  [TurtleBot3 Burger Manual](http://www.robotis.com/service/download.php?no=748&_gl=1*wc3mgb*_gcl_au*MjA1NzQ5Nzg5OS4xNzc0NjY5NDE3), and follow online instructions for [OpenCR setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/opencr_setup/#opencr-setup) if needed.

## Networking and Teleoperation
Code typically runs on both the TurtleBot3 and in the Dev Container of an external computer.
This section is needed to ensure messages can be sent to and from both machines.

1. The dev container is set to using CycloneDDS, so first, we need to either [install it on the TurtleBot3](#cyclonedds), or [change the DDS to FastDDS in the container](#fastdds).
2. Set the `ROS_DOMAIN_ID` in the container to be the same as that in the TurtleBot3 in `~/.bashrc`:
For example, make sure the following are in the `~/.bashrc` of both the TurtleBot3 and the container:
```bash
export ROS_DOMAIN_ID=8
```
3. On both the container running on your computer and the TurtleBot3, stop the ROS2 daemon, then source `~/.bashrc`
```bash
ros2 daemon stop
exec bash
```
4. Try using keyboard teleoperation as described by the [official instructions](https://emanual.robotis.com/docs/en/platform/turtlebot3/basic_operation/#basic-operation) (with the robot)

### FastDDS
Do this in the container on your computer (and not the robot).
1. In the container's `~/.bashrc` file, change the `RMW_IMPLEMENTATION` as follows:
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### CycloneDDS 
This is often the preferred DDS over wifi. 
If FastDDS does not work well, try this.

Do this on the robot (and not your computer or container).
1. On the TurtleBot3, install CycloneDDS
```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

1. On the TurtleBot3, add these lines to `~/.bashrc`:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=~/cyclonedds.xml
```

3. Run `ifconfig` and note the network interface through which the TurtleBot3 will connect to the same network as the computer running the container. 
For example, it may be something like `wlan0`.
Note this interface name down.

4. Edit `~/cyclonedds.xml` (e.g. `nano ~/cyclonedds.xml`) and paste the following into the file, ensuring that the `NetworkInterface name` is set to the network interface noted above.
```xml
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="wlan0"/>
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>none</ParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
```

## SLAM and Navigation
1. To make a map, follow instructions in the [official docs](https://emanual.robotis.com/docs/en/platform/turtlebot3/slam/#run-slam-node).
2. To Navigate using Nav2, also follow the [official docs](https://emanual.robotis.com/docs/en/platform/turtlebot3/navigation/#run-navigation-nodes).
Nav2 comes with a set of built-in planners, controllers, and more.
3. It is actually unnecessary to use everything in Nav2, as we also have our own planner and controller.
After starting `navigation2.launch.py` and estimating the initial robot pose, we can run our own planner and controller by launching our own launch file.
```bash
ros2 launch controller controller.launch.py
```
Take care to set up the map and planner and controller parameters correctly, and be ready to pick up the TurtleBot3 or `^C` just in case!