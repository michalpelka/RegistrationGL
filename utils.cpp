#include "utils.h"
#include <sophus/se3.hpp>
#include "navsat.h"
void my_utils::saveState(const std::string &fn, const std::vector<Eigen::Matrix4d>& trajectory, const std::map<int,Eigen::Matrix4d>& gt_icp_resutls)
{
    const auto serialize = [](Sophus::SE3d &m){
        std::stringstream ss;
        const auto m_l = m.log();
        ss << m_l[0] << " "<< m_l[1] << " "<< m_l[2] << " ";
        ss << m_l[3] << " "<< m_l[4] << " "<< m_l[5];
        return ss.str();
    };

    boost::property_tree::ptree  pt;
    pt.put("trajectory.count", trajectory.size());
    for (int i =0; i < trajectory.size(); i++)
    {
        auto t = Sophus::SE3d::fitToSE3(trajectory[i].matrix());
        pt.put("trajectory.keyframes."+std::to_string(i), serialize(t));
    }

    for (const auto &x : gt_icp_resutls)
    {
        auto t = Sophus::SE3d::fitToSE3(x.second);
        pt.put("gt_icp.icp."+std::to_string(x.first), serialize(t));
    }

    boost::property_tree::write_json(fn, pt);
}

void my_utils::LoadState(const std::string &fn, std::vector<Eigen::Matrix4d>& trajectory, std::map<int,Eigen::Matrix4d>& gt_icp_resutls)
{
    boost::property_tree::ptree  pt;
    boost::property_tree::read_json(fn, pt);
    const auto deserialize = [](const std::string str){
        std::cout << "str " << str << std::endl;
        std::stringstream ss(str);
        Sophus::Vector6d l;
        ss >> l[0];ss >> l[1];ss >> l[2];
        ss >> l[3];ss >> l[4];ss >> l[5];
        return Sophus::SE3d::exp(l);
    };

    for (int i =0; ; i++)
    {
        auto data = pt.get_optional<std::string>("trajectory.keyframes."+std::to_string(i));
        if (data) {
            std::string str;
            auto t = deserialize(*data);
            trajectory.push_back(t.matrix());
        }else{
            break;
        }
    }
//    const auto & pt_icp = pt.get_child("gt_icp.icp");
//
//    for (auto &it : pt_icp){
//        const int id=std::atoi(it.first.data()) ;
//        const auto mat = deserialize(it.second.data());
//        gt_icp_resutls[id] = mat.matrix();
//    }


}

std::pair<double, Eigen::Matrix4d> my_utils::loadLineCsv(std::string line){
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

std::vector<float> my_utils::loadTXTCloud(const std::string &fn){
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

Eigen::Matrix4d my_utils::loadMat(const std::string& fn){
    Eigen::Matrix4d m;
    std::ifstream ifss(fn);
    for (int i =0; i < 16; i++) {
        ifss >> m.data()[i];
    }
    ifss.close();
    std::cout << m.transpose() << std::endl;
    return m.transpose();;
}
void my_utils::saveMat(const std::string& fn, const Eigen::Matrix4d& mat){
    Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, " ", "\n", "", "", "", "");
    std::ofstream fmt_ofs (fn);
    fmt_ofs << mat.format(HeavyFmt);
    fmt_ofs.close();
}
Eigen::Matrix4d my_utils::orthogonize(const Eigen::Matrix4d & p )
{
    Eigen::Matrix4d ret = p;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(ret.block<3,3>(0,0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    double d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
    Eigen::Matrix3d diag = Eigen::Matrix3d::Identity() * d;
    ret.block<3,3>(0,0) = svd.matrixU() * diag * svd.matrixV().transpose();
    return ret;
}
 std::vector<std::string> my_utils::glob(const std::string& pat){
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

Eigen::Vector2d my_utils::loadNovatel(const std::string &fn){
    std::fstream infile(fn);
    std::string line;
    std::vector<double> lat;
    std::vector<double> lon;
    while (std::getline(infile, line)){
        std::replace( line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        std::vector<std::string> fields;
        while (!ss.eof()){
            std::string f;
            ss >> f;
            fields.push_back(f);
        }
        if (fields.size()>12){
            lat.push_back(std::atof(fields[11].c_str()));
            lon.push_back(std::atof(fields[12].c_str()));
        }
    }
    assert(lat.size()==lon.size());
    if (!lat.empty()) {

        float lat_mean = std::accumulate(lat.begin(), lat.end(), 0.f) / lat.size();
        float lon_mean = std::accumulate(lon.begin(), lon.end(), 0.f) / lon.size();
        Eigen::Vector2d utm;
        std::string zone;
        double gamma;
        RobotLocalization::NavsatConversions::LLtoUTM(lat_mean, lon_mean, utm[0], utm[1], zone, gamma);
        std::cout << "Zone " << zone << " gamma " << gamma << std::endl;
        return utm;
    }
    return {-1,-1};
}