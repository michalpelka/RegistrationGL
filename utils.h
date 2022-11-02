#pragma once
#include <Eigen/Dense>
#include <glob.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <glob.h>
#include "navsat.h"
namespace my_utils{

    void saveState(const std::string &fn, const std::vector<Eigen::Matrix4d>& trajectory, const std::map<int,Eigen::Matrix4d>& gt_icp_resutls);

    void LoadState(const std::string &fn, std::vector<Eigen::Matrix4d>& trajectory, std::map<int,Eigen::Matrix4d>& gt_icp_resutls);

    std::pair<double, Eigen::Matrix4d> loadLineCsv(std::string line);

    std::vector<float> loadTXTCloud(const std::string &fn);

    Eigen::Vector2d loadNovatel(const std::string &fn);

    Eigen::Vector4d loadGround(const std::string &fn);

    Eigen::Matrix4d loadMat(const std::string& fn);

    void saveMat(const std::string& fn, const Eigen::Matrix4d& mat);

    Eigen::Matrix4d orthogonize(const Eigen::Matrix4d & p );

    Eigen::Matrix4d get4by4FromPlane(const Eigen::Vector4d& plane, Eigen::Vector2d pointXY);

    std::vector<std::string> glob(const std::string& pat);

}