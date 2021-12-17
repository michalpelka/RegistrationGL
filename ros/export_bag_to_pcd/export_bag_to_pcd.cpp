//
// Created by michal on 19.12.2020.
//

//boost
#include <boost/program_options.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

//ros
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <boost/foreach.hpp>
#include <fstream>
#include <pcl/common/transforms.h>

void saveMat(const std::string& fn, const Eigen::Matrix4d& mat){
    Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, " ", "\n", "", "", "", "");
    std::ofstream fmt_ofs (fn);
    fmt_ofs << mat.format(HeavyFmt);
    fmt_ofs.close();
}


int main(int argc, char *argv[]) {

    rosbag::Bag bag;

    Eigen::Affine3d laser_offset{Eigen::Affine3d::Identity()};
    laser_offset.translation() = Eigen::Vector3d{0.2,0.0,0.5};
    auto Q = Eigen::Quaterniond{ 0.96593,0.0, 0.0, -0.25882};
    Q.normalize();
    laser_offset.rotate(Q);
    //laser_offset = laser_offset.inverse();
    std::vector<std::string>bag_files{
        "/media/michal/ext/garaz2/2021-12-13-01-25-20.bag"
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-18-56_0.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-19-05_1.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-19-32_0.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-19-41_1.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-19-50_2.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-19-59_3.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-20-07_4.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-20-16_5.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-20-26_6.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-20-35_7.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-20-45_8.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-20-55_9.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-21-04_10.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-21-14_11.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-21-23_12.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-21-33_13.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-21-42_14.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-21-51_15.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-22-00_16.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-22-09_17.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-22-19_18.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-22-29_19.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-22-39_20.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-22-48_21.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-22-58_22.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-23-07_23.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-23-17_24.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-23-26_25.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-23-36_26.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-23-46_27.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-23-55_28.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-24-05_29.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-24-15_30.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-24-24_31.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-24-34_32.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-24-43_33.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-24-53_34.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-25-02_35.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-25-11_36.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-25-21_37.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-25-30_38.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-25-40_39.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-25-49_40.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-25-59_41.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-26-08_42.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-26-17_43.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-26-27_44.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-26-36_45.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-26-45_46.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-26-55_47.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-27-04_48.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-27-13_49.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-27-23_50.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-27-32_51.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-27-42_52.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-27-52_53.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-28-02_54.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-28-11_55.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-28-21_56.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-28-31_57.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-28-40_58.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-28-50_59.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-28-59_60.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-29-08_61.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-29-18_62.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-29-27_63.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-29-37_64.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-29-46_65.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-29-55_66.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-30-05_67.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-30-14_68.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-30-24_69.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-30-33_70.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-30-43_71.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-30-52_72.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-31-02_73.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-31-11_74.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-31-21_75.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-31-30_76.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-31-40_77.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-31-49_78.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-31-58_79.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-32-06_80.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-32-15_81.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-32-23_82.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-32-32_83.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-32-41_84.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-32-51_85.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-33-01_86.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-33-10_87.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-33-19_88.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-33-27_89.bag",
//            "/mnt/540C28560C283580/ENRICH/mobile_mapping/2021-10-07-16-33-36_90.bag"
    };




    Eigen::Affine3d old_affine(Eigen::Matrix4d::Zero());
    Eigen::Matrix4d last_added(Eigen::Matrix4d::Zero());

    ros::Time last_odom;
    Eigen::Affine3d start_integration(Eigen::Matrix4d::Zero());
    pcl::PointCloud<pcl::PointXYZI> aggregate;
    int count = 0;
    for (std::string &bag_file:bag_files) {

        std::cout << "bag files " << bag_file << std::endl;
        bag.open(bag_file, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        const std::string pointcloud_topic{"/velodyne_points"};
        topics.push_back(pointcloud_topic);
        topics.push_back(std::string("/odometry/filtered"));
        rosbag::View view(bag, rosbag::TopicQuery(topics));
        BOOST_FOREACH(rosbag::MessageInstance const m, view) {
                //std::cout << m.getTopic() << std::endl;
                nav_msgs::Odometry::ConstPtr s = m.instantiate<nav_msgs::Odometry>();
                if (s) {
                    //std::cout << s->pose << std::endl;

                    Eigen::Vector3d t{s->pose.pose.position.x, s->pose.pose.position.y,
                                      s->pose.pose.position.z};
                    Eigen::Quaterniond q{s->pose.pose.orientation.w,
                                         s->pose.pose.orientation.x, s->pose.pose.orientation.y,
                                         s->pose.pose.orientation.z};
                    Eigen::Affine3d pose_eigen(Eigen::Affine3d::Identity());

                    pose_eigen.translate(t);
                    pose_eigen.rotate(q);
                    old_affine = pose_eigen*laser_offset;
                }
                sensor_msgs::PointCloud2::ConstPtr s2 = m.instantiate<sensor_msgs::PointCloud2>();
                double ts = m.getTime().toSec();

                if (s2 && m.getTopic() == pointcloud_topic) {
                    if (old_affine.matrix() == Eigen::Matrix4d::Zero()) continue;
                    if (start_integration.matrix() == Eigen::Matrix4d::Zero()){
                        std::cout << "start aggregating"<<std::endl;
                        start_integration = old_affine;
                    }
                    double dist = (old_affine.translation() - Eigen::Affine3d(start_integration).translation()).norm();
                    double dist_ang = Eigen::Quaterniond(old_affine.rotation()).angularDistance(Eigen::Quaterniond(start_integration.rotation()));
                    pcl::PointCloud<pcl::PointXYZI> f1;
                    pcl::PointCloud<pcl::PointXYZI> f2;

                    pcl::fromROSMsg(*s2, f1);
                    pcl::transformPointCloud(f1,f2, (start_integration.inverse()*old_affine).cast<float>().matrix());
                    aggregate+= f2;
                    std::cout << "dist " << dist << " dist_ang " << dist_ang << std::endl;
                    if (dist > 0.25 ){
                        char prefix[128];
                        snprintf(prefix, 128, "%04d", count);
                        pcl::PointCloud<pcl::PointXYZI> aggregate_sub;
                        pcl::VoxelGrid<pcl::PointXYZI> sor;
                        sor.setInputCloud (aggregate.makeShared());
                        sor.setLeafSize (0.05f, 0.05f, 0.05f);
                        sor.filter (aggregate_sub);
                        pcl::io::savePCDFileBinary("//media/michal/ext/garaz2/scans/cloud_"+std::string(prefix)+".pcd", aggregate_sub);
                        saveMat("//media/michal/ext/garaz2/scans/cloud_"+std::string(prefix)+".txt", start_integration.matrix());
                        start_integration = Eigen::Affine3d(Eigen::Matrix4d::Zero());
                        aggregate.clear();
                        count++;
                    }
                }
            }
        bag.close();
    }

}
