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

namespace my_utils{




    std::pair<double, Eigen::Matrix4d> loadLineCsv(std::string line){
        std::replace_if(std::begin(line), std::end(line),
                        [](std::string::value_type v) { return v==','; },
                        ' ');
         std::stringstream ss(line);
         double ts;
         Eigen::Matrix4d matrix(Eigen::Matrix4d::Identity());
         ss >> ts;
         for (int i =0; i < 12; i ++)
         {
             ss >> matrix.data()[i];
         }
         //std::cout << "matrix.transpose() "<< matrix.transpose() << std::endl;
         return std::make_pair(ts, matrix.transpose());
    }

    std::vector<float> loadTXTCloud(const std::string &fn){
        std::vector<float> ret;
        ret.reserve(1e6);
        std::fstream infile(fn);
        std::string line;
        while (std::getline(infile, line)){
            float x,y,z,i,ts;
            std::stringstream ss(line);
            ss >> x;
            ss >> y;
            ss >> z;
            ss >> i;
            ss >> ts;

            ret.push_back(x);
            ret.push_back(y);
            ret.push_back(z);
            ret.push_back(i);
            ret.push_back(ts);
        }
        return ret;
    }



Eigen::Matrix4d loadMat(const std::string& fn){
        Eigen::Matrix4d m;
        std::ifstream ifss(fn);
        for (int i =0; i < 16; i++) {
            ifss >> m.data()[i];
        }
        ifss.close();
        std::cout << m.transpose() << std::endl;
        return m.transpose();;
    }
    void saveMat(const std::string& fn, const Eigen::Matrix4d& mat){
        Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, " ", "\n", "", "", "", "");
        std::ofstream fmt_ofs (fn);
        fmt_ofs << mat.format(HeavyFmt);
        fmt_ofs.close();
    }
    Eigen::Matrix4d orthogonize(const Eigen::Matrix4d & p )
    {
        Eigen::Matrix4d ret = p;
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(ret.block<3,3>(0,0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        double d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
        Eigen::Matrix3d diag = Eigen::Matrix3d::Identity() * d;
        ret.block<3,3>(0,0) = svd.matrixU() * diag * svd.matrixV().transpose();
        return ret;
    }

    inline std::vector<std::string> glob(const std::string& pat){
        using namespace std;
        glob_t glob_result;
        glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
        vector<string> ret;
        for(unsigned int i=0;i<glob_result.gl_pathc;++i){
            ret.push_back(string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
        return ret;
    }
}